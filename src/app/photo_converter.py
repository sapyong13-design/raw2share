import os
import rawpy
from io import BytesIO
from PIL import Image, ImageFile
from app.autocorrect import apply_autocorrect, correct_raw_corner_shading, get_smart_auto_decision, save_smart_auto_contact_sheet
from app.adjustment_engine import AdjustmentSettings, apply_adjustments, calculate_resize
from app.metadata import copy_exif_metadata
from app.image_analysis import analyze_image
from app.raw_engine import (
    RawEngine,
    build_darktable_command,
    build_rawtherapee_command,
    detect_raw_engines,
    parse_raw_engine,
    run_external_raw_engine,
    select_raw_engine,
)

ImageFile.MAXBLOCK = 2**26  # 64 MB

def _is_smart_auto_mode(mode: str) -> bool:
    mode_clean = (mode or "").strip().lower()
    return "smart" in mode_clean or "auto" in mode_clean

def _append_smart_auto_metadata(message_parts: list[str], rgb_img, is_raw: bool, settings: dict, output_path: str) -> None:
    mode = settings.get('autocorrect_mode', 'Off')
    if not _is_smart_auto_mode(mode):
        return
    decision = get_smart_auto_decision(rgb_img, is_raw=is_raw, batch_context=settings.get('batch_context'), smart_strength=settings.get('smart_auto_strength', 'Event Balanced'))
    top = ", ".join(f"{name}:{score:.1f}" for name, score in decision.get('candidates', [])[:3])
    message_parts.append(f"Smart Auto: {decision.get('profile')} ({decision.get('score', 0):.1f}).")
    if top:
        message_parts.append(f"Top candidates: {top}.")
    if settings.get('save_contact_sheet', True):
        import os
        base, _ = os.path.splitext(output_path)
        sheet_path = f"{base}_smartauto_candidates.jpg"
        save_smart_auto_contact_sheet(rgb_img, sheet_path, is_raw=is_raw, batch_context=settings.get('batch_context'), smart_strength=settings.get('smart_auto_strength', 'Event Balanced'))
        message_parts.append(f"Candidates: {os.path.basename(sheet_path)}.")
    if settings.get('save_ai_debug', False):
        from app.region_detection import detect_regions, generate_region_debug_sheet
        from PIL import Image
        import os
        base, _ = os.path.splitext(output_path)
        debug_sheet_path = f"{base}_ai_debug.jpg"
        masks = detect_regions(rgb_img, is_raw=is_raw)
        debug_sheet_np = generate_region_debug_sheet(rgb_img, masks)
        debug_sheet_pil = Image.fromarray(debug_sheet_np)
        debug_sheet_pil.save(debug_sheet_path, "JPEG", quality=85)
        message_parts.append(f"AI Debug: {os.path.basename(debug_sheet_path)}.")

def _image_is_too_overexposed(img_pil: Image.Image) -> bool:
    """Detect camera previews that are too clipped to use as RAW base."""
    import numpy as np

    small = img_pil.convert('RGB').resize((512, 512))
    arr = np.asarray(small).astype(np.float32)
    luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    clipped_ratio = float(np.mean(luminance >= 250.0))
    mean_luminance = float(np.mean(luminance))
    p95 = float(np.percentile(luminance, 95))

    # Outdoor camera previews can be heavily blown out. If the preview is already
    # clipped this much, RAW processing is safer because it may retain highlights.
    return clipped_ratio > 0.08 or (mean_luminance > 178.0 and p95 > 248.0)

def convert_raw_to_jpg(input_path: str, output_path: str, settings: dict) -> dict:
    """
    Converts a CR3/CR2 raw photo to JPG based on settings.
    
    settings dict keys:
    - quality: int (90-100, default 98)
    - keep_resolution: bool (default True)
    - autocorrect_mode: str ('Off', 'Natural', 'Bright', 'Vivid', 'Low Light')
    - use_camera_wb: bool (default True)
    - copy_exif: bool (default True)
    
    Returns:
        dict: {'success': bool, 'message': str, 'size_before': int, 'size_after': int}
    """
    try:
        if not os.path.exists(input_path):
            return {
                'success': False,
                'message': f"Input file not found: {input_path}",
                'size_before': 0,
                'size_after': 0
            }

        size_before = os.path.getsize(input_path)
        photo_reasoning = ""
        smart_auto_messages = []
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Extract EXIF metadata stats and add to batch_context for Autocorrect V4
        from app.autocorrect import extract_metadata_stats
        meta_stats = extract_metadata_stats(input_path)
        if 'batch_context' not in settings or settings['batch_context'] is None:
            settings['batch_context'] = {}
        settings['batch_context']['metadata'] = meta_stats

        _, ext = os.path.splitext(input_path.lower())
        if ext in ('.jpg', '.jpeg'):
            # Load standard image directly
            img_pil = Image.open(input_path)
            # Apply autocorrect (requires converting to numpy array then back to PIL)
            autocorrect_mode = settings.get('autocorrect_mode', 'Off')
            if autocorrect_mode != 'Off':
                import numpy as np
                rgb_img = np.array(img_pil.convert('RGB'))
                analysis = analyze_image(rgb_img, is_raw=False)
                photo_reasoning = analysis.reasoning
                _append_smart_auto_metadata(smart_auto_messages, rgb_img, False, settings, output_path)
                rgb_img = apply_autocorrect(rgb_img, autocorrect_mode, is_raw=False, batch_context=settings.get('batch_context'), smart_strength=settings.get('smart_auto_strength', 'Event Balanced'))
                img_pil = Image.fromarray(rgb_img)
        else:
            autocorrect_mode = settings.get('autocorrect_mode', 'Off')
            requested_engine = parse_raw_engine(settings.get('raw_engine', RawEngine.AUTO.value))
            detected_engines = detect_raw_engines(
                settings.get('rawtherapee_path'),
                settings.get('darktable_path'),
            )
            used_embedded_preview = False
            used_external_engine = False
            raw_engine_used = "Unknown"
            raw_scene = "Unknown"
            embedded = None
            has_safe_preview = False

            with rawpy.imread(input_path) as raw:
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        embedded = Image.open(BytesIO(thumb.data)).convert('RGB')
                        has_safe_preview = (
                            embedded.width >= 3000
                            and embedded.height >= 2000
                            and not _image_is_too_overexposed(embedded)
                        )
                except Exception:
                    embedded = None
                    has_safe_preview = False

                selected_engine = select_raw_engine(
                    requested_engine,
                    has_safe_preview=has_safe_preview,
                    rawtherapee_path=detected_engines.rawtherapee,
                    darktable_path=detected_engines.darktable,
                )

                if selected_engine == RawEngine.CAMERA_PREVIEW and embedded is not None:
                    img_pil = embedded
                    raw_engine_used = "Camera Preview"
                    used_embedded_preview = True
                elif selected_engine in (RawEngine.RAWTHERAPEE, RawEngine.DARKTABLE):
                    if selected_engine == RawEngine.RAWTHERAPEE and detected_engines.rawtherapee:
                        cmd = build_rawtherapee_command(detected_engines.rawtherapee, input_path, output_path, settings.get('quality', 92))
                    else:
                        cmd = build_darktable_command(detected_engines.darktable, input_path, output_path, settings.get('quality', 92))
                    ok, external_msg = run_external_raw_engine(cmd)
                    if ok and os.path.exists(output_path):
                        used_external_engine = True
                        raw_engine_used = selected_engine.value
                        img_pil = Image.open(output_path).convert('RGB')
                    else:
                        use_camera_wb = settings.get('use_camera_wb', True)
                        rgb_img = raw.postprocess(use_camera_wb=use_camera_wb, no_auto_bright=False, output_bps=8)
                        raw_engine_used = "LibRaw/rawpy fallback"
                        if settings.get('correct_corner_shading', True):
                            meta_stats = settings.get('batch_context', {}).get('metadata') if settings.get('batch_context') else None
                            rgb_img = correct_raw_corner_shading(rgb_img, metadata=meta_stats)
                        analysis = analyze_image(rgb_img, is_raw=True)
                        raw_scene = analysis.scene
                        photo_reasoning = analysis.reasoning
                        _append_smart_auto_metadata(smart_auto_messages, rgb_img, True, settings, output_path)
                        rgb_img = apply_autocorrect(rgb_img, autocorrect_mode, is_raw=True, batch_context=settings.get('batch_context'), smart_strength=settings.get('smart_auto_strength', 'Event Balanced'))
                        img_pil = Image.fromarray(rgb_img)
                else:
                    use_camera_wb = settings.get('use_camera_wb', True)
                    rgb_img = raw.postprocess(use_camera_wb=use_camera_wb, no_auto_bright=False, output_bps=8)
                    raw_engine_used = "LibRaw/rawpy"
                    if settings.get('correct_corner_shading', True):
                        meta_stats = settings.get('batch_context', {}).get('metadata') if settings.get('batch_context') else None
                        rgb_img = correct_raw_corner_shading(rgb_img, metadata=meta_stats)
                    analysis = analyze_image(rgb_img, is_raw=True)
                    raw_scene = analysis.scene
                    photo_reasoning = analysis.reasoning
                    _append_smart_auto_metadata(smart_auto_messages, rgb_img, True, settings, output_path)
                    rgb_img = apply_autocorrect(rgb_img, autocorrect_mode, is_raw=True, batch_context=settings.get('batch_context'), smart_strength=settings.get('smart_auto_strength', 'Event Balanced'))
                    img_pil = Image.fromarray(rgb_img)

            if used_embedded_preview:
                import numpy as np
                rgb_img = np.array(img_pil.convert('RGB'))
                if settings.get('correct_corner_shading', True):
                    meta_stats = settings.get('batch_context', {}).get('metadata') if settings.get('batch_context') else None
                    rgb_img = correct_raw_corner_shading(rgb_img, strength=0.65, metadata=meta_stats)
                analysis = analyze_image(rgb_img, is_raw=False)
                raw_scene = analysis.scene
                photo_reasoning = analysis.reasoning
                if autocorrect_mode != 'Off':
                    _append_smart_auto_metadata(smart_auto_messages, rgb_img, False, settings, output_path)
                    rgb_img = apply_autocorrect(rgb_img, autocorrect_mode, is_raw=False, batch_context=settings.get('batch_context'), smart_strength=settings.get('smart_auto_strength', 'Event Balanced'))
                img_pil = Image.fromarray(rgb_img)
            elif used_external_engine:
                analysis = analyze_image(np.array(img_pil.convert('RGB')), is_raw=True)
                raw_scene = analysis.scene
                photo_reasoning = analysis.reasoning
            elif 'rgb_img' in locals() and raw_scene == "Unknown":
                analysis = analyze_image(rgb_img, is_raw=True)
                raw_scene = analysis.scene
                photo_reasoning = analysis.reasoning

        manual_adjustments = settings.get('manual_adjustments')
        if manual_adjustments:
            import numpy as np
            adjustment_settings = AdjustmentSettings(**manual_adjustments)
            rgb_img = np.array(img_pil.convert('RGB'))
            rgb_img = apply_adjustments(rgb_img, adjustment_settings)
            img_pil = Image.fromarray(rgb_img)

        keep_res = settings.get('keep_resolution', True)
        max_dimension = settings.get('max_dimension', 0)
        if not keep_res:
            max_dimension = max_dimension or 3840
        if max_dimension and max_dimension > 0:
            w, h = img_pil.size
            new_w, new_h = calculate_resize(w, h, int(max_dimension))
            if (new_w, new_h) != (w, h):
                img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Save to JPG
        quality = settings.get('quality', 98)
        try:
            img_pil.save(
                output_path,
                "JPEG",
                quality=quality,
                optimize=True
            )
        except OSError as e:
            if "broken data stream" in str(e):
                img_pil.save(
                    output_path,
                    "JPEG",
                    quality=quality
                )
            else:
                raise e

        message = "Photo converted successfully."
        if 'raw_engine_used' in locals() and raw_engine_used != "Unknown":
            message += f" RAW engine: {raw_engine_used}."
        if 'raw_scene' in locals() and raw_scene != "Unknown":
            message += f" Scene: {raw_scene}."
        if photo_reasoning:
            message += f" Analysis: {photo_reasoning}"
        if smart_auto_messages:
            message += " " + " ".join(smart_auto_messages)
        batch_profile = settings.get('batch_profile')
        if batch_profile:
            message += f" Batch consistency: {batch_profile}."
        if 'used_embedded_preview' in locals() and used_embedded_preview:
            message += " Used embedded full-size camera JPEG preview as RAW base."
        if 'used_external_engine' in locals() and used_external_engine:
            message += " Used external RAW engine."

        # Copy EXIF if requested
        if settings.get('copy_exif', True):
            meta_ok, meta_msg = copy_exif_metadata(input_path, output_path)
            if not meta_ok:
                message += f" ({meta_msg})"

        size_after = os.path.getsize(output_path)

        return {
            'success': True,
            'message': message,
            'size_before': size_before,
            'size_after': size_after
        }

    except Exception as e:
        return {
            'success': False,
            'message': f"Error: {str(e)}",
            'size_before': os.path.getsize(input_path) if os.path.exists(input_path) else 0,
            'size_after': 0
        }
