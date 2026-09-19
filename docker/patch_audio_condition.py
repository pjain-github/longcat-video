#!/usr/bin/env python3
"""Add an audio-embedding strength control to LongCat's single-avatar demo."""

from pathlib import Path
import sys


target = Path(sys.argv[1])
source = target.read_text()

argument = '''    parser.add_argument(
        '--audio_condition_scale',
        type=float,
        default=1.0,
        help='Scale audio embeddings before generation. Use 0.0-1.0; lower values soften audio-driven motion.',
    )
'''
argument_marker = '''    parser.add_argument(
        '--stage_1',
'''
if "--audio_condition_scale" not in source:
    if argument_marker not in source:
        raise SystemExit("Could not find the argument insertion point in upstream demo")
    source = source.replace(argument_marker, argument + argument_marker, 1)

assignment = "    audio_guidance_scale = args.audio_guidance_scale\n"
replacement = '''    audio_guidance_scale = args.audio_guidance_scale
    audio_condition_scale = args.audio_condition_scale
    if not 0.0 <= audio_condition_scale <= 1.0:
        raise ValueError("--audio_condition_scale must be between 0.0 and 1.0")
'''
if "audio_condition_scale = args.audio_condition_scale" not in source:
    if assignment not in source:
        raise SystemExit("Could not find the audio-guidance assignment in upstream demo")
    source = source.replace(assignment, replacement, 1)

embedding_marker = '''        if torch.isnan(full_audio_emb).any():
            raise ValueError(f"broken audio embedding with nan values")
'''
embedding_replacement = '''        if torch.isnan(full_audio_emb).any():
            raise ValueError(f"broken audio embedding with nan values")
        if audio_condition_scale != 1.0:
            full_audio_emb = full_audio_emb * audio_condition_scale
            print(f"[INFO] Audio conditioning scale: {audio_condition_scale:.2f}")
'''
if "[INFO] Audio conditioning scale:" not in source:
    if embedding_marker not in source:
        raise SystemExit("Could not find the audio-embedding insertion point in upstream demo")
    source = source.replace(embedding_marker, embedding_replacement, 1)

target.write_text(source)
