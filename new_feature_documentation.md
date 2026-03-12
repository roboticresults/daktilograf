"""
New Feature: Real-time Segment Processing
-----------------------------------------

This version introduces support for real-time processing of transcription
segments via the `new_segment_callback` parameter.

How it works:
- The `transcribe_audio` function now accepts an optional
  `new_segment_callback` argument.
- When provided, each recognized segment is passed to the callback,
  allowing immediate typing and logging.
- The callback receives a `Segment` object with `.text`, `.start`, and
  `.end` attributes.
- Typical callback actions include:
  * Typing the segment text via `pyautogui` with natural timing.
  * Logging the text to a file for record-keeping.
- If no callback is provided, the original behavior (transcribe the
  entire recording at once) is preserved.

Example callback:
    def _real_time_callback(segment):
        if segment.text and len(segment.text) > 2:
            type_text(segment.text, typing_interval)
            log_transcription(segment.text)

The callback is automatically invoked by the transcription loop when
`new_segment_callback` is supplied.

For more information, see the documentation of the `transcribe_audio`
function and the `run_dictation_loop` function.
"""