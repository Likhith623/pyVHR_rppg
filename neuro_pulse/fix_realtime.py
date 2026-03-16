import os

path = '/Users/likhith./pyVHR_rppg/neuro_pulse/src/realtime_pipeline.py'
with open(path, 'r') as f:
    text = f.read()

# Make buffer smaller so it responds faster
text = text.replace("MAXLEN = 300  # 10 seconds at 30 FPS", "MAXLEN = 150  # 5 seconds at 30 FPS")

old_init = '''    frame_count = 0
    t_start = time.time()
    last_roi: Dict[str, float] = {"forehead": 0.0, "left_cheek": 0.0, "right_cheek": 0.0}'''
new_init = '''    frame_count = 0
    missing_face_frames = 0
    t_start = time.time()
    last_roi: Dict[str, float] = {"forehead": 0.0, "left_cheek": 0.0, "right_cheek": 0.0}'''
text = text.replace(old_init, new_init)

old_else = '''                else:
                    # No face detected
                    green_buffer.append(green_buffer[-1] if green_buffer else 0.0)'''
new_else = '''                else:
                    # No face detected
                    missing_face_frames += 1
                    green_buffer.append(green_buffer[-1] if green_buffer else 0.0)'''
text = text.replace(old_else, new_else)

old_succ = '''                    if roi_vals is not None:
                        green_buffer.append(roi_vals["combined"])'''
new_succ = '''                    missing_face_frames = 0
                    if roi_vals is not None:
                        green_buffer.append(roi_vals["combined"])'''
text = text.replace(old_succ, new_succ)

old_overlay = '''                    overlay_text(
                        frame, "No face detected", (10, 30),
                        color=(0, 165, 255)
                    )'''
new_overlay = '''                    overlay_text(
                        frame, "No face detected", (10, 30),
                        color=(0, 165, 255)
                    )
                    
                    if missing_face_frames > 15:
                        green_buffer.clear()
                        for k in roi_buffers:
                            roi_buffers[k].clear()
                        last_result = None
                        verdict_history.clear()'''
text = text.replace(old_overlay, new_overlay)

print("success:", text != old_init)

with open(path, 'w') as f:
    f.write(text)
