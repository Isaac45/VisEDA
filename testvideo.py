from __future__ import annotations
import argparse, tempfile, time
from pathlib import Path
import numpy as np
from viseda.video import VideoEDA


def section(t):
    print('\n' + '═' * 62 + f'\n  {t}\n' + '═' * 62)


def ok(m): print(f'  ✔  {m}')


def info(m): print(f'  ℹ  {m}')


def warn(m): print(f'  ⚠  {m}')


def make_synthetic_video(T=60, H=96, W=128, seed=0, scene='moving'):
    rng = np.random.default_rng(seed);
    video = np.zeros((T, H, W, 3), dtype=np.uint8)
    for t in range(T):
        f = np.zeros((H, W, 3), dtype=np.uint8) + 25
        if scene == 'flicker':
            f[:] = 40 + int(100 * (0.5 + 0.5 * np.sin(t / 4)))
        elif scene == 'blurred':
            x = int((W - 40) * t / max(T - 1, 1));
            f[H // 2 - 8:H // 2 + 8, x:x + 40] = [80, 180, 250];
            f = ((f.astype(float) + np.roll(f, 4, 1) + np.roll(f, -4, 1)) / 3).astype(np.uint8)
        elif scene == 'random':
            f = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        else:
            x = int((W - 24) * t / max(T - 1, 1));
            f[H // 3:H // 3 + 24, x:x + 24] = [220, 80, 60]
        f = np.clip(f + rng.integers(0, 8, f.shape, dtype=np.uint8), 0, 255)
        video[t] = f
    return video


def write_video(path: Path, video, fps=20.0):
    try:
        import cv2
    except ImportError:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*('mp4v' if path.suffix.lower() == '.mp4' else 'XVID'))
    writer = cv2.VideoWriter(str(path), fourcc, fps, (video.shape[2], video.shape[1]))
    if not writer.isOpened(): return False
    for frame in video:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release();
    return path.exists()


def write_dataset(root: Path, n=6):
    labels = ['traffic', 'sports', 'surveillance'];
    scenes = ['moving', 'flicker', 'blurred']
    ok_all = True
    for i in range(n):
        ok_all = write_video(root / labels[i % 3] / f'video_{i:03d}.mp4',
                             make_synthetic_video(T=45 + i * 3, seed=i, scene=scenes[i % 3]), fps=20 + i) and ok_all
    return ok_all


def save_or_show(eda, method, save, out_dir, **kwargs):
    if save:
        out_dir.mkdir(exist_ok=True)
        path = str(out_dir / f'{method}.png');
        kwargs['save_path'] = path
        getattr(eda, method)(**kwargs);
        ok(f'Saved → {path}')
    else:
        getattr(eda, method)(**kwargs)


def test_arrays(quick):
    section('1 · LOAD — Video arrays')
    scenes = ['moving', 'flicker', 'blurred', 'random'];
    labels = ['traffic', 'lighting', 'motion', 'noise']
    videos = [make_synthetic_video(seed=i, scene=scenes[i % 4]) for i in range(8)]
    eda = VideoEDA(verbose=True, frame_sample_rate=4 if quick else 1, max_frames_per_video=80)
    t = time.perf_counter();
    eda.load_arrays(videos, labels=[labels[i % 4] for i in range(8)], fps=24.0)
    assert len(eda._records) == 8 and sum(not r.is_corrupt for r in eda._records) == 8
    ok(f'Loaded 8 videos in {time.perf_counter() - t:.2f}s');
    return eda


def test_summary(eda):
    section('2 · SUMMARY')
    s = eda.summary();
    inv = s['inventory'];
    assert inv['valid_videos'] > 0
    ok(f"Total: {inv['total_videos']} Valid: {inv['valid_videos']} Corrupt: {inv['corrupt_videos']}")
    ok(f"Format distribution: {inv['format_distribution']}");
    ok(f"Label distribution: {inv['label_distribution']}")
    ok(f"Mean frame count: {s['temporal']['frame_count'].get('mean')}")
    ok(f"Mean sharpness: {s['quality']['sharpness_mean'].get('mean')}")
    ok(f"Motion blur fraction: {s['motion']['motion_blur_fraction'].get('mean')}")


def test_fields(eda):
    section('3 · PER-RECORD FIELD VALIDATION')
    req = ['n_frames', 'sampled_frames', 'fps', 'duration_sec', 'height', 'width', 'brightness_mean', 'contrast_mean',
           'sharpness_mean', 'blur_fraction', 'motion_blur_fraction', 'frame_diff_mean', 'scene_change_count',
           'temporal_brightness_std', 'rgb_mean', 'rgb_std']
    for r in [r for r in eda._records if not r.is_corrupt][:5]:
        for f in req: assert getattr(r, f) is not None, f'{f} missing on {r.path}'
    ok(f'All {len(req)} fields populated')


def test_temporal(eda):
    section('4 · TEMPORAL PROFILE')
    p = eda.temporal_profile(0);
    assert len(p['brightness']) == eda.get_record(0).sampled_frames
    ok(f"Temporal profile length: {len(p['brightness'])}")


def test_pairwise(eda):
    section('5 · PAIRWISE VIDEO DISTANCES')
    D, n = eda.pairwise_video_distances();
    assert D.shape[0] == D.shape[1] == len(n)
    ok(f'Distance matrix computed: {D.shape}')


def test_plots(eda, save, out_dir):
    section('6 · PLOTS')
    save_or_show(eda, 'plot_dataset', save, out_dir);
    ok('Dataset dashboard rendered')
    save_or_show(eda, 'plot', save, out_dir, video_index=0);
    ok('Single video dashboard rendered')
    save_or_show(eda, 'plot_videos_grid', save, out_dir, n=6);
    ok('Video grid rendered')
    save_or_show(eda, 'plot_motion_profile', save, out_dir, video_index=0);
    ok('Motion profile rendered')
    save_or_show(eda, 'plot_pairwise_video_distances', save, out_dir);
    ok('Pairwise heatmap rendered')


def test_report(eda, report):
    section('7 · HTML REPORT')
    p = eda.report(report);
    assert Path(p).exists();
    ok(f'Report saved → {p}')


def test_directory(tmp, quick):
    section('8 · LOAD — Directory of video files')
    if not write_dataset(tmp): info('OpenCV VideoWriter unavailable — skipping'); return None
    eda = VideoEDA(verbose=True, frame_sample_rate=4 if quick else 1);
    eda.load(tmp, label_from_parent=True)
    assert sum(not r.is_corrupt for r in eda._records) == 6;
    ok('Loaded 6 videos from directory');
    return eda


def test_formats(tmp, quick):
    section('9 · FILE FORMATS — mp4 and avi')
    v = make_synthetic_video(T=30);
    tmp.mkdir(parents=True, exist_ok=True)
    if not (write_video(tmp / 'sample.mp4', v) or write_video(tmp / 'sample.avi', v)): info(
        'OpenCV VideoWriter unavailable — skipping'); return
    eda = VideoEDA(verbose=False, frame_sample_rate=2 if quick else 1);
    eda.load(tmp)
    assert any(not r.is_corrupt for r in eda._records);
    ok('Loaded written video files')


def test_edges():
    section('10 · EDGE CASES')
    try:
        VideoEDA(verbose=False).summary(); assert False
    except RuntimeError:
        ok('RuntimeError raised before load')
    eda = VideoEDA(verbose=False, max_videos=2)
    eda.load_arrays([make_synthetic_video(seed=i) for i in range(5)])
    assert len(eda._records) == 2
    ok('max_videos respected')
    eda2 = VideoEDA(verbose=False)
    eda2.load_arrays([make_synthetic_video(T=1)])
    assert eda2.summary()['motion']['frame_diff_mean']['mean'] == 0
    ok('Single-frame video handled')
    eda3 = VideoEDA(verbose=False)
    eda3.load_arrays([np.array([np.nan])])
    assert eda3.summary()['inventory']['corrupt_videos'] == 1
    ok('Invalid video marked corrupt')


def test_user_file(path, quick, save, out, report):
    section('11 · USER-SUPPLIED VIDEO FILE')
    eda = VideoEDA(verbose=True, frame_sample_rate=5 if quick else 1)
    eda.load(path)
    if eda.summary()['inventory']['valid_videos'] == 0: warn('No valid video loaded'); return
    test_summary(eda)
    # Generate the final report from the user-supplied video file, not the synthetic test data.
    eda.report(report)
    ok(f'Report saved from user file → {report}')
    if save: save_or_show(eda, 'plot', save, out, video_index=0)


def test_user_dir(path, quick, save, out, report):
    section('12 · USER-SUPPLIED DIRECTORY')
    eda = VideoEDA(verbose=True, frame_sample_rate=5 if quick else 1)
    eda.load(path, label_from_parent=True)
    if eda.summary()['inventory']['valid_videos'] == 0: warn('No valid videos loaded'); return
    test_summary(eda)
    # Generate the final report from the user-supplied directory, not the synthetic test data.
    eda.report(report)
    ok(f'Report saved from user directory → {report}')
    if save: save_or_show(eda, 'plot_dataset', save, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file')
    ap.add_argument('--dir')
    ap.add_argument('--save-plots', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--report', default='viseda_video_report.html')
    args = ap.parse_args()
    out = Path('outputs_video')
    passed = failed = 0
    print('╔════════════════════════════════════════════════════════════╗')
    print('║  VisEDA — VideoEDA Complete Test Suite                     ║')
    print('╚════════════════════════════════════════════════════════════╝')
    print(
        f'\n  Save plots : {args.save_plots}\n  Quick mode : {args.quick}\n  File       : {args.file}\n  Directory  : {args.dir}')

    def run(name, fn):
        nonlocal passed, failed
        try:
            fn(); passed += 1
        except Exception:
            failed += 1
            import traceback
            print(f'\n  ✘ TEST FAILED: {name}')
            traceback.print_exc()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        eda = test_arrays(args.quick)
        passed += 1
        for name, fn in [('summary', lambda: test_summary(eda)), ('fields', lambda: test_fields(eda)),
                         ('temporal', lambda: test_temporal(eda)), ('pairwise', lambda: test_pairwise(eda)),
                         ('plots', lambda: test_plots(eda, args.save_plots, out)),
                         ('report', lambda: test_report(eda, args.report)), ('directory', lambda: (
            lambda e: test_summary(e) if e else None)(test_directory(tmp / 'videos', args.quick))),
                         ('formats', lambda: test_formats(tmp / 'formats', args.quick)), ('edges', test_edges)]: run(
            name, fn)
        if args.file: run('user_file', lambda: test_user_file(args.file, args.quick, args.save_plots, out, args.report))
        if args.dir: run('user_directory', lambda: test_user_dir(args.dir, args.quick, args.save_plots, out, args.report))
    print('\n' + '═' * 62)
    print(f'  Results: {passed} passed | {failed} failed')
    print(f'  Report: {args.report}')
    print('═' * 62)


if __name__ == '__main__': main()
