from __future__ import annotations

import html, math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError as exc:
        raise ImportError('VideoEDA requires OpenCV for video files: pip install opencv-python') from exc


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _stat(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(count=0, mean=None, std=None, min=None, p25=None, median=None, p75=None, max=None)
    return dict(count=int(a.size), mean=round(float(a.mean()), 6), std=round(float(a.std()), 6),
                min=round(float(a.min()), 6), p25=round(float(np.percentile(a,25)), 6),
                median=round(float(np.median(a)), 6), p75=round(float(np.percentile(a,75)), 6),
                max=round(float(a.max()), 6))


class VideoRecord:
    fields = [
        'path','label','file_ext','file_size_kb','is_corrupt','error','n_frames','sampled_frames','fps','duration_sec',
        'height','width','channels','aspect_ratio','megapixels','brightness_mean','brightness_std','contrast_mean',
        'contrast_std','sharpness_mean','sharpness_std','sharpness_min','blur_fraction','frame_diff_mean','frame_diff_std',
        'motion_intensity_mean','motion_intensity_std','motion_blur_fraction','scene_change_count','scene_change_rate',
        'temporal_brightness_std','temporal_contrast_std','rgb_mean','rgb_std','sample_frames'
    ]
    def __init__(self):
        for f in self.fields:
            setattr(self, f, None)
        self.is_corrupt = False
        self.error = None


class VideoEDA:
    SUPPORTED_EXTS = {'.mp4','.avi','.mov','.mkv','.webm','.mpeg','.mpg','.m4v'}

    def __init__(self, verbose: bool=True, max_videos: Optional[int]=None, frame_sample_rate: int=5,
                 max_frames_per_video: int=300, blur_threshold: float=80.0,
                 motion_blur_threshold: float=80.0, scene_change_threshold: float=35.0,
                 resize_width: Optional[int]=320):
        self.verbose = verbose
        self.max_videos = max_videos
        self.frame_sample_rate = max(1, int(frame_sample_rate))
        self.max_frames_per_video = max(1, int(max_frames_per_video))
        self.blur_threshold = float(blur_threshold)
        self.motion_blur_threshold = float(motion_blur_threshold)
        self.scene_change_threshold = float(scene_change_threshold)
        self.resize_width = resize_width
        self._records = []
        self._arrays = {}
        self._label_map = {}
        self._loaded = False
        self._results = {}

    def load(self, source: Union[str, Path, Sequence[Union[str, Path]]], labels: Optional[Dict[str,str]]=None,
             label_from_parent: bool=False, recursive: bool=True):
        paths = self._resolve_paths(source, recursive)
        if self.max_videos is not None:
            paths = paths[:self.max_videos]
        if labels:
            self._label_map = {str(Path(k).resolve()): v for k, v in labels.items()}
        self._records, self._arrays = [], {}
        self._log(f'Found {len(paths)} video file(s) — computing statistics …')
        for i, p in enumerate(paths):
            self._log(f'  [{i+1}/{len(paths)}] {p.name}')
            self._records.append(self._analyse_file(p, label_from_parent))
        self._loaded = True
        self._log(f'Done. {len(self._records)} video(s) loaded ({sum(r.is_corrupt for r in self._records)} corrupt).')
        return self

    def load_arrays(self, videos: Sequence[np.ndarray], labels: Optional[Sequence[str]]=None,
                    names: Optional[Sequence[str]]=None, fps: Union[float, Sequence[float]]=30.0):
        if self.max_videos is not None:
            videos = videos[:self.max_videos]
            labels = labels[:self.max_videos] if labels is not None else None
            names = names[:self.max_videos] if names is not None else None
        self._records, self._arrays = [], {}
        self._log(f'Loading {len(videos)} video array(s) …')
        for i, arr in enumerate(videos):
            r = VideoRecord()
            r.path = names[i] if names and i < len(names) else f'<array_{i}>'
            r.label = labels[i] if labels and i < len(labels) else None
            r.file_ext = 'array'
            try:
                fpsi = float(fps[i]) if isinstance(fps, (list, tuple, np.ndarray)) else float(fps)
                video = self._normalise_video_array(arr)
                frames = self._sample_frames_from_array(video)
                self._arrays[r.path] = frames
                self._fill_stats(r, frames, video.shape[0], fpsi)
            except Exception as e:
                r.is_corrupt, r.error = True, str(e)
                self._log(f'  ✗ {r.path}: {e}')
            self._records.append(r)
        self._loaded = True
        return self

    def summary(self):
        self._check_loaded()
        valid = [r for r in self._records if not r.is_corrupt]
        corrupt = [r for r in self._records if r.is_corrupt]
        if not valid:
            out = {'inventory': {'total_videos': len(self._records), 'valid_videos': 0, 'corrupt_videos': len(corrupt),
                                 'corrupt_paths': [r.path for r in corrupt], 'format_distribution': {}, 'label_distribution': None},
                   'spatial': {}, 'temporal': {}, 'quality': {}, 'motion': {}, 'colour': {},
                   'labels': {'label_distribution': None, 'class_imbalance_ratio': None}, 'error': 'No valid videos found.'}
            self._results['summary'] = out
            return out
        def arr(x): return [getattr(r, x) for r in valid if getattr(r, x) is not None]
        labels = [r.label for r in valid if r.label]
        label_dist = dict(Counter(labels)) if labels else None
        rgb_means = [r.rgb_mean for r in valid if r.rgb_mean is not None]
        rgb_stds = [r.rgb_std for r in valid if r.rgb_std is not None]
        out = {
            'inventory': {'total_videos': len(self._records), 'valid_videos': len(valid), 'corrupt_videos': len(corrupt),
                          'corrupt_paths': [r.path for r in corrupt], 'format_distribution': dict(Counter(r.file_ext for r in valid)),
                          'label_distribution': label_dist, 'file_size_kb': _stat(arr('file_size_kb'))},
            'spatial': {'height': _stat(arr('height')), 'width': _stat(arr('width')), 'aspect_ratio': _stat(arr('aspect_ratio')), 'megapixels': _stat(arr('megapixels'))},
            'temporal': {'frame_count': _stat(arr('n_frames')), 'sampled_frames': _stat(arr('sampled_frames')), 'fps': _stat(arr('fps')), 'duration_sec': _stat(arr('duration_sec')), 'temporal_brightness_std': _stat(arr('temporal_brightness_std')), 'temporal_contrast_std': _stat(arr('temporal_contrast_std'))},
            'quality': {'brightness_mean': _stat(arr('brightness_mean')), 'brightness_std': _stat(arr('brightness_std')), 'contrast_mean': _stat(arr('contrast_mean')), 'sharpness_mean': _stat(arr('sharpness_mean')), 'sharpness_min': _stat(arr('sharpness_min')), 'blur_fraction': _stat(arr('blur_fraction'))},
            'motion': {'frame_diff_mean': _stat(arr('frame_diff_mean')), 'motion_intensity_mean': _stat(arr('motion_intensity_mean')), 'motion_blur_fraction': _stat(arr('motion_blur_fraction')), 'scene_change_count': _stat(arr('scene_change_count')), 'scene_change_rate': _stat(arr('scene_change_rate'))},
            'colour': {'rgb_mean_mean': np.vstack(rgb_means).mean(axis=0).round(6).tolist() if rgb_means else None,
                       'rgb_std_mean': np.vstack(rgb_stds).mean(axis=0).round(6).tolist() if rgb_stds else None},
            'labels': {'label_distribution': label_dist, 'class_imbalance_ratio': round(max(label_dist.values()) / max(min(label_dist.values()), 1), 6) if label_dist and len(label_dist)>1 else None}
        }
        self._results['summary'] = out
        return out

    def get_record(self, index=0):
        self._check_loaded(); return self._records[index]

    def get_frames(self, index=0):
        self._check_loaded(); r = self._records[index]
        if r.is_corrupt: raise ValueError(r.error)
        return self._arrays[r.path]

    def temporal_profile(self, index=0):
        frames = self.get_frames(index)
        gray = self._gray(frames)
        sharp = np.asarray([self._lap_var(g) for g in gray])
        return {'brightness': gray.mean(axis=(1,2)), 'contrast': gray.std(axis=(1,2)), 'sharpness': sharp, 'frame_diff': self._diffs(gray)}

    def pairwise_video_distances(self, max_videos=50):
        self._check_loaded(); valid = [r for r in self._records if not r.is_corrupt][:max_videos]
        if len(valid) < 2: raise ValueError('Need at least two valid videos.')
        X, names = [], []
        for r in valid:
            X.append([r.n_frames or 0, r.fps or 0, r.duration_sec or 0, r.height or 0, r.width or 0, r.brightness_mean or 0, r.contrast_mean or 0, r.sharpness_mean or 0, r.blur_fraction or 0, r.motion_blur_fraction or 0, r.motion_intensity_mean or 0, r.scene_change_rate or 0])
            names.append(r.label or Path(str(r.path)).stem)
        X = np.nan_to_num(np.asarray(X, float)); sd = X.std(axis=0); sd[sd==0] = 1
        X = (X - X.mean(axis=0)) / sd
        return np.sqrt(((X[:,None,:] - X[None,:,:])**2).sum(axis=2)), names

    def plot_dataset(self, figsize=(24, 24), save_path=None, dpi=150):
        """Comprehensive dataset-level dashboard for all loaded videos."""
        self._check_loaded()
        valid = [r for r in self._records if not r.is_corrupt]
        if not valid:
            raise RuntimeError('No valid videos to plot.')
        plt = _plt()
        import matplotlib as mpl
        fig = plt.figure(figsize=figsize, facecolor='white')
        fig.suptitle('VideoEDA — Dataset Analysis', fontsize=20, fontweight='bold', y=0.995)
        gs = mpl.gridspec.GridSpec(6, 4, figure=fig, hspace=0.55, wspace=0.35,
                                   left=0.05, right=0.98, top=0.96, bottom=0.03)

        def ax(pos):
            a = fig.add_subplot(pos)
            a.set_facecolor('#f6f8fa')
            for sp in a.spines.values():
                sp.set_edgecolor('#d0d7de')
            a.tick_params(colors='#57606a', labelsize=8)
            return a

        def values(attr):
            return [getattr(r, attr) for r in valid if getattr(r, attr) is not None]

        # Row 0: dataset overview and label distribution
        self._plot_dataset_overview(ax(gs[0, :2]), valid)
        self._plot_label_dist_horizontal(ax(gs[0, 2:]), valid)

        # Row 1: spatial and temporal inventory
        self._plot_hist(ax(gs[1, 0]), values('n_frames'), 'Frame Count')
        self._plot_hist(ax(gs[1, 1]), values('duration_sec'), 'Duration (seconds)')
        self._plot_hist(ax(gs[1, 2]), values('fps'), 'FPS')
        self._plot_hist(ax(gs[1, 3]), values('aspect_ratio'), 'Aspect Ratio')

        # Row 2: quality metrics
        self._plot_hist(ax(gs[2, 0]), values('brightness_mean'), 'Brightness Mean')
        self._plot_hist(ax(gs[2, 1]), values('contrast_mean'), 'Contrast Mean')
        self._plot_hist(ax(gs[2, 2]), values('sharpness_mean'), 'Sharpness Mean')
        self._plot_hist(ax(gs[2, 3]), values('blur_fraction'), 'Blur Fraction')

        # Row 3: motion and temporal consistency
        self._plot_hist(ax(gs[3, 0]), values('motion_intensity_mean'), 'Motion Intensity')
        self._plot_hist(ax(gs[3, 1]), values('motion_blur_fraction'), 'Motion Blur Fraction')
        self._plot_hist(ax(gs[3, 2]), values('scene_change_rate'), 'Scene Change Rate')
        self._plot_hist(ax(gs[3, 3]), values('temporal_brightness_std'), 'Temporal Brightness Std')

        # Row 4: relationships and colour
        self._plot_motion_sharpness_scatter(ax(gs[4, 0]), valid)
        self._plot_duration_motion_scatter(ax(gs[4, 1]), valid)
        self._plot_rgb_mean_distribution(ax(gs[4, 2]), valid)
        self._plot_format_distribution(ax(gs[4, 3]), valid)

        # Row 5: previews, pairwise diversity, resolution, quality bars
        self._plot_frame_preview_strip(ax(gs[5, 0]), valid)
        self._plot_pairwise_heatmap(ax(gs[5, 1]), valid)
        self._plot_resolution_scatter(ax(gs[5, 2]), valid)
        self._plot_quality_bars(ax(gs[5, 3]), valid)

        self._finalise(fig, save_path, dpi)

    def plot(self, video_index=0, figsize=(16,10), save_path=None, dpi=150):
        frames=self.get_frames(video_index); prof=self.temporal_profile(video_index); plt=_plt(); fig,ax=plt.subplots(2,3,figsize=figsize); ax=ax.ravel(); fig.suptitle('VideoEDA — Single Video',fontweight='bold')
        ax[0].imshow(frames[len(frames)//2]); ax[0].set_title('Middle Frame'); ax[0].axis('off')
        for a,k,t in [(ax[1],'brightness','Brightness'),(ax[2],'contrast','Contrast'),(ax[3],'sharpness','Sharpness'),(ax[4],'frame_diff','Frame Difference')]: a.plot(prof[k]); a.set_title(t)
        self._hist(ax[5], prof['sharpness'], 'Sharpness Distribution'); self._finalise(fig, save_path, dpi)

    def plot_videos_grid(self, n=12, cols=4, save_path=None, dpi=150):
        idxs=[i for i,r in enumerate(self._records) if not r.is_corrupt][:n]
        plt=_plt(); rows=math.ceil(len(idxs)/cols); fig,axes=plt.subplots(rows,cols,figsize=(cols*4,rows*3),squeeze=False)
        for a in axes.ravel(): a.axis('off')
        for a,i in zip(axes.ravel(),idxs):
            f=self.get_frames(i); a.imshow(f[len(f)//2]); a.set_title(self._records[i].label or Path(str(self._records[i].path)).stem,fontsize=8)
        self._finalise(fig,save_path,dpi)

    def plot_motion_profile(self, video_index=0, save_path=None, dpi=150):
        p=self.temporal_profile(video_index); plt=_plt(); fig,ax=plt.subplots(figsize=(12,5)); ax.plot(p['frame_diff'],label='Frame diff'); ax.plot(p['sharpness'][1:]/(p['sharpness'].max()+1e-9)*max(p['frame_diff'].max() if len(p['frame_diff']) else 1,1),label='Sharpness scaled'); ax.legend(); ax.set_title('Motion/Sharpness Profile'); self._finalise(fig,save_path,dpi)

    def plot_pairwise_video_distances(self, save_path=None, dpi=150):
        D,names=self.pairwise_video_distances(); plt=_plt(); fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(D); plt.colorbar(im,ax=ax); ax.set_title('Pairwise Video Distances'); self._finalise(fig,save_path,dpi)

    def plot_temporal_statistics(self, video_index=0, save_path=None, dpi=150):
        """Detailed temporal statistics for one video: brightness, contrast, sharpness and motion."""
        p = self.temporal_profile(video_index)
        plt = _plt()
        fig, ax = plt.subplots(2, 2, figsize=(16, 10), facecolor='white')
        ax = ax.ravel()
        for a, key, title in [
            (ax[0], 'brightness', 'Brightness over sampled frames'),
            (ax[1], 'contrast', 'Contrast over sampled frames'),
            (ax[2], 'sharpness', 'Sharpness over sampled frames'),
            (ax[3], 'frame_diff', 'Frame difference / motion'),
        ]:
            self._style_axis(a)
            a.plot(p[key])
            a.set_title(title)
            a.set_xlabel('Sampled frame index')
        fig.suptitle('VideoEDA — Temporal Statistics', fontsize=15, fontweight='bold')
        self._finalise(fig, save_path, dpi)

    def plot_quality_summary(self, save_path=None, dpi=150):
        """Dataset-level video quality summary."""
        self._check_loaded()
        valid = [r for r in self._records if not r.is_corrupt]
        if not valid:
            raise RuntimeError('No valid videos to plot.')
        plt = _plt()
        fig, ax = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')
        ax = ax.ravel()
        self._plot_hist(ax[0], [r.brightness_mean for r in valid], 'Brightness Mean')
        self._plot_hist(ax[1], [r.contrast_mean for r in valid], 'Contrast Mean')
        self._plot_hist(ax[2], [r.sharpness_mean for r in valid], 'Sharpness Mean')
        self._plot_hist(ax[3], [r.blur_fraction for r in valid], 'Blur Fraction')
        self._plot_hist(ax[4], [r.motion_blur_fraction for r in valid], 'Motion Blur Fraction')
        self._plot_motion_sharpness_scatter(ax[5], valid)
        fig.suptitle('VideoEDA — Quality Summary', fontsize=15, fontweight='bold')
        self._finalise(fig, save_path, dpi)

    def plot_frame_samples(self, video_index=0, n=12, cols=4, save_path=None, dpi=150):
        """Grid of sampled frames from one video."""
        frames = self.get_frames(video_index)
        if len(frames) == 0:
            raise RuntimeError('No sampled frames available.')
        plt = _plt()
        n = min(n, len(frames))
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False, facecolor='white')
        for a in axes.ravel():
            a.axis('off')
        idx = np.linspace(0, len(frames) - 1, n).astype(int)
        for a, i in zip(axes.ravel(), idx):
            a.imshow(frames[i])
            a.set_title(f'Frame {i}', fontsize=8)
        fig.suptitle('VideoEDA — Sampled Frames', fontsize=14, fontweight='bold')
        self._finalise(fig, save_path, dpi)

    def report(self, output_path='viseda_video_report.html'):
        _generate_html_report(self.summary(), output_path); self._log(f'Report saved → {output_path}'); return output_path

    def _analyse_file(self,p,label_from_parent):
        r=VideoRecord(); r.path=str(p); r.file_ext=p.suffix.lower(); r.file_size_kb=p.stat().st_size/1024 if p.exists() else None; r.label=p.parent.name if label_from_parent else self._label_map.get(str(p.resolve()))
        try:
            frames,n,fps=self._read_file(p); self._arrays[r.path]=frames; self._fill_stats(r,frames,n,fps)
        except Exception as e:
            r.is_corrupt=True; r.error=str(e); self._log(f'  ✗ {p.name}: {e}')
        return r

    def _fill_stats(self,r,frames,n,fps):
        frames=self._normalise_video_array(frames); T,H,W,C=frames.shape
        r.n_frames=int(n); r.sampled_frames=int(T); r.fps=float(fps) if fps else None; r.duration_sec=float(n/fps) if fps else None; r.height=H; r.width=W; r.channels=C; r.aspect_ratio=W/H; r.megapixels=H*W/1e6
        gray=self._gray(frames); bright=gray.mean(axis=(1,2)); cont=gray.std(axis=(1,2)); sharp=np.asarray([self._lap_var(g) for g in gray]); dif=self._diffs(gray)
        r.brightness_mean=float(bright.mean()); r.brightness_std=float(bright.std()); r.contrast_mean=float(cont.mean()); r.contrast_std=float(cont.std()); r.sharpness_mean=float(sharp.mean()); r.sharpness_std=float(sharp.std()); r.sharpness_min=float(sharp.min()); r.blur_fraction=float(np.mean(sharp<self.blur_threshold)); r.temporal_brightness_std=float(bright.std()); r.temporal_contrast_std=float(cont.std())
        if len(dif):
            r.frame_diff_mean=float(dif.mean()); r.frame_diff_std=float(dif.std()); r.motion_intensity_mean=r.frame_diff_mean; r.motion_intensity_std=r.frame_diff_std; r.scene_change_count=int(np.sum(dif>self.scene_change_threshold)); r.scene_change_rate=float(r.scene_change_count/len(dif)); r.motion_blur_fraction=float(np.mean((dif>np.percentile(dif,75)) & (sharp[1:]<self.motion_blur_threshold)))
        else:
            r.frame_diff_mean=r.frame_diff_std=r.motion_intensity_mean=r.motion_intensity_std=0.0; r.scene_change_count=0; r.scene_change_rate=0.0; r.motion_blur_fraction=0.0
        rgb=frames.astype(np.float32)/255.0; r.rgb_mean=rgb.reshape(-1,3).mean(axis=0); r.rgb_std=rgb.reshape(-1,3).std(axis=0); r.sample_frames=frames[np.linspace(0,T-1,min(T,6)).astype(int)]

    def _read_file(self,p):
        cv2=_cv2(); cap=cv2.VideoCapture(str(p));
        if not cap.isOpened(): raise ValueError('Could not open video')
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); fps=float(cap.get(cv2.CAP_PROP_FPS) or 30.0); frames=[]; i=0
        while True:
            ok,fr=cap.read();
            if not ok: break
            if i%self.frame_sample_rate==0:
                fr=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB); frames.append(self._resize(fr))
                if len(frames)>=self.max_frames_per_video: break
            i+=1
        cap.release();
        if not frames: raise ValueError('No readable frames found')
        return np.stack(frames), n or i, fps or 30.0

    def _normalise_video_array(self,a):
        a=np.asarray(a)
        if a.ndim==3: a=a[:,:,:,None]
        if a.ndim!=4: raise ValueError('Video arrays must be (T,H,W) or (T,H,W,C)')
        if a.shape[0]<1: raise ValueError('Video has no frames')
        if a.shape[-1]==1: a=np.repeat(a,3,axis=-1)
        if a.shape[-1]>3: a=a[...,:3]
        if not np.all(np.isfinite(a)): raise ValueError('Video contains NaN or infinite values')
        if a.dtype!=np.uint8:
            a=a.astype(np.float32); a=a*255 if a.max()<=1.5 else a; a=np.clip(a,0,255).astype(np.uint8)
        return a

    def _sample_frames_from_array(self,v):
        idx=np.arange(0,v.shape[0],self.frame_sample_rate)[:self.max_frames_per_video]
        return np.stack([self._resize(f) for f in v[idx]])
    def _resize(self,f):
        if self.resize_width is None or f.shape[1] <= self.resize_width: return f
        cv2=_cv2(); h,w=f.shape[:2]; nw=self.resize_width; nh=max(1,int(h*nw/w)); return cv2.resize(f,(nw,nh),interpolation=cv2.INTER_AREA)
    def _gray(self,f): return .299*f[...,0].astype(float)+.587*f[...,1].astype(float)+.114*f[...,2].astype(float)
    def _lap_var(self,g):
        try: return float(_cv2().Laplacian(g.astype(np.float32), _cv2().CV_32F).var())
        except Exception: gy,gx=np.gradient(g.astype(float)); return float((gx*gx+gy*gy).var())
    def _diffs(self,g): return np.mean(np.abs(np.diff(g.astype(float),axis=0)),axis=(1,2)) if len(g)>1 else np.asarray([])
    def _resolve_paths(self,source,recursive):
        if isinstance(source,(list,tuple)): paths=[Path(x) for x in source]
        else:
            p=Path(source)
            if p.is_dir(): paths=[x for x in (p.rglob('*') if recursive else p.glob('*')) if x.is_file()]
            elif p.is_file(): paths=[p]
            else: raise FileNotFoundError(source)
        return sorted([p for p in paths if p.suffix.lower() in self.SUPPORTED_EXTS])
    def _check_loaded(self):
        if not self._loaded: raise RuntimeError('No videos loaded. Call load() or load_arrays() first.')
    def _log(self,msg):
        if self.verbose: print(f'[viseda] {msg}')
    def _plot_hist(self, ax, data, title):
        """Compatibility wrapper used by the comprehensive dashboard."""
        return self._hist(ax, data, title)

    def _hist(self,ax,data,title):
        a=np.asarray([x for x in data if x is not None and np.isfinite(x)],float); ax.set_title(title)
        if len(a): ax.hist(a,bins=min(20,max(5,len(a)))); ax.axvline(a.mean(),ls='--')
    def _label_bar(self,ax,records):
        labels=[r.label for r in records if r.label]; ax.set_title('Labels')
        if labels:
            k,v=zip(*Counter(labels).items()); ax.barh(k,v)
    def _style_ax(self, ax):
        """Compatibility wrapper for older plot helper calls."""
        return self._style_axis(ax)

    def _style_axis(self, ax):
        ax.set_facecolor('#f6f8fa')
        for sp in ax.spines.values():
            sp.set_edgecolor('#d0d7de')
        ax.tick_params(colors='#57606a', labelsize=8)

    def _plot_dataset_overview(self, ax, records):
        ax.axis('off')
        total = len(self._records)
        valid = len(records)
        corrupt = sum(r.is_corrupt for r in self._records)
        labels = [r.label for r in records if r.label]
        mean_duration = np.nanmean([r.duration_sec for r in records if r.duration_sec is not None]) if records else 0
        mean_fps = np.nanmean([r.fps for r in records if r.fps is not None]) if records else 0
        mean_frames = np.nanmean([r.n_frames for r in records if r.n_frames is not None]) if records else 0
        mean_motion_blur = np.nanmean([r.motion_blur_fraction for r in records if r.motion_blur_fraction is not None]) if records else 0
        lines = [
            f'Total videos:       {total}',
            f'Valid videos:       {valid}',
            f'Corrupt videos:     {corrupt}',
            f'Unique labels:      {len(set(labels)) if labels else 0}',
            f'Mean frames:        {mean_frames:.1f}',
            f'Mean FPS:           {mean_fps:.2f}',
            f'Mean duration:      {mean_duration:.2f}s',
            f'Mean motion blur:   {mean_motion_blur:.4f}',
        ]
        ax.text(0.04, 0.94, '\n'.join(lines), va='top', ha='left', transform=ax.transAxes,
                fontsize=9, family='monospace',
                bbox=dict(boxstyle='round,pad=0.45', facecolor='#eaeef2', edgecolor='#d0d7de'))
        ax.set_title('Dataset Overview')

    def _plot_label_dist_horizontal(self, ax, records):
        self._style_axis(ax)
        labels = [r.label for r in records if r.label]
        ax.set_title('Label Distribution')
        if not labels:
            ax.text(0.5, 0.5, 'No labels\n(use label_from_parent=True)', ha='center', va='center')
            return
        names, counts = zip(*Counter(labels).most_common(20))
        y = np.arange(len(names))
        ax.barh(y, counts)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Count')
        ax.invert_yaxis()

    def _plot_motion_sharpness_scatter(self, ax, records):
        self._style_axis(ax)
        x = [r.motion_intensity_mean for r in records if r.motion_intensity_mean is not None and r.sharpness_mean is not None]
        y = [r.sharpness_mean for r in records if r.motion_intensity_mean is not None and r.sharpness_mean is not None]
        ax.scatter(x, y, alpha=0.75)
        ax.set_title('Motion vs Sharpness')
        ax.set_xlabel('Motion intensity')
        ax.set_ylabel('Sharpness')

    def _plot_duration_motion_scatter(self, ax, records):
        self._style_axis(ax)
        x = [r.duration_sec for r in records if r.duration_sec is not None and r.motion_intensity_mean is not None]
        y = [r.motion_intensity_mean for r in records if r.duration_sec is not None and r.motion_intensity_mean is not None]
        ax.scatter(x, y, alpha=0.75)
        ax.set_title('Duration vs Motion')
        ax.set_xlabel('Duration (s)')
        ax.set_ylabel('Motion intensity')

    def _plot_rgb_mean_distribution(self, ax, records):
        self._style_axis(ax)
        vals = [r.rgb_mean for r in records if r.rgb_mean is not None]
        ax.set_title('RGB Mean Distribution')
        if not vals:
            ax.text(0.5, 0.5, 'No colour data', ha='center', va='center')
            return
        arr = np.vstack(vals)
        ax.hist(arr[:, 0], alpha=0.45, label='Red')
        ax.hist(arr[:, 1], alpha=0.45, label='Green')
        ax.hist(arr[:, 2], alpha=0.45, label='Blue')
        ax.legend(fontsize=7)
        ax.set_xlabel('Mean channel value (0–1)')

    def _plot_format_distribution(self, ax, records):
        self._style_axis(ax)
        counts = Counter(r.file_ext for r in records)
        ax.set_title('File Format Distribution')
        if counts:
            ax.bar(list(counts.keys()), list(counts.values()))
            ax.set_ylabel('Count')

    def _plot_frame_preview_strip(self, ax, records):
        ax.axis('off')
        chosen = []
        for rec in records[:6]:
            try:
                frames = self._arrays.get(rec.path)
                if frames is not None and len(frames):
                    chosen.append(frames[len(frames)//2])
            except Exception:
                pass
        ax.set_title('Representative Frames')
        if not chosen:
            ax.text(0.5, 0.5, 'No preview frames', ha='center', va='center')
            return
        min_h = min(f.shape[0] for f in chosen)
        thumbs = []
        for f in chosen:
            if f.shape[0] != min_h:
                step = max(1, f.shape[0] // min_h)
                f = f[::step][:min_h]
            thumbs.append(f[:min_h])
        montage = np.concatenate(thumbs, axis=1)
        ax.imshow(montage)

    def _plot_pairwise_heatmap(self, ax, records):
        self._style_axis(ax)
        ax.set_title('Pairwise Video Distances')
        try:
            D, names = self.pairwise_video_distances(max_videos=min(30, len(records)))
            im = ax.imshow(D, aspect='auto')
            if len(names) <= 12:
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names, fontsize=6)
        except Exception as exc:
            ax.text(0.5, 0.5, f'Unavailable\n{exc}', ha='center', va='center')

    def _plot_resolution_scatter(self, ax, records):
        self._style_axis(ax)
        x = [r.width for r in records if r.width is not None and r.height is not None]
        y = [r.height for r in records if r.width is not None and r.height is not None]
        ax.scatter(x, y, alpha=0.75)
        ax.set_title('Resolution Scatter')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')

    def _plot_quality_bars(self, ax, records):
        self._style_axis(ax)
        labels = ['Blur', 'Motion blur', 'Scene change']
        vals = [
            np.nanmean([r.blur_fraction for r in records if r.blur_fraction is not None]),
            np.nanmean([r.motion_blur_fraction for r in records if r.motion_blur_fraction is not None]),
            np.nanmean([r.scene_change_rate for r in records if r.scene_change_rate is not None]),
        ]
        vals = [0 if not np.isfinite(v) else v for v in vals]
        ax.bar(labels, vals)
        ax.set_title('Quality Flags / Rates')
        ax.set_ylim(bottom=0)

    def _finalise(self,fig,save_path,dpi):
        # plt=_plt(); fig.tight_layout()
        plt = _plt()
        try:
            fig.set_constrained_layout(True)
        except Exception:
            pass

        fig.subplots_adjust(
            left=0.04,
            right=0.98,
            top=0.95,
            bottom=0.04,
            hspace=0.45,
            wspace=0.30
        )
        if save_path: Path(save_path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(save_path,dpi=dpi,bbox_inches='tight'); plt.close(fig)
        else: plt.show()

def _fmt(v):
    if v is None: return 'N/A'
    if isinstance(v,float): return f'{v:.4f}'
    return html.escape(str(v))
def _card(title,stats):
    rows=''.join(f'<div class="stat"><span>{k}</span><span class="val">{_fmt(stats.get(k))}</span></div>' for k in ['min','max','mean','median','std','p25','p75'] if stats and k in stats)
    return f'<div class="card"><h3>{html.escape(title)}</h3>{rows or "No data"}</div>'
def _bar(title,d):
    if not d: return f'<div class="card"><h3>{title}</h3>No data</div>'
    m=max(d.values()) or 1; rows=''.join(f'<div class="bar-row"><span class="bar-label">{html.escape(str(k))}</span><div class="bar"><div class="bar-fill" style="width:{100*v/m:.1f}%"></div></div><span class="bar-count">{v}</span></div>' for k,v in d.items())
    return f'<div class="card"><h3>{title}</h3>{rows}</div>'
def _generate_html_report(s,output_path):
    inv=s.get('inventory',{}); sp=s.get('spatial',{}); tm=s.get('temporal',{}); q=s.get('quality',{}); mo=s.get('motion',{})
    css='body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:2rem;color:#1f2328}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:.9rem}.card{background:#f6f8fa;border:1px solid #d0d7de;border-radius:8px;padding:1rem}.stat{display:flex;justify-content:space-between;border-bottom:1px solid #d0d7de;padding:.18rem 0}.val{color:#58a6ff}.bar-row{display:flex;gap:.4rem;margin:.2rem 0}.bar-label{width:120px}.bar{flex:1;background:#d0d7de;border-radius:3px;height:9px}.bar-fill{height:100%;background:#58a6ff}.bar-count{width:45px;text-align:right;color:#58a6ff}h2{color:#58a6ff;margin-top:1.8rem}h3{color:#57606a;text-transform:uppercase;font-size:.78rem}'
    html_doc=f'<!doctype html><html><head><meta charset="utf-8"><title>VisEDA Video Report</title><style>{css}</style></head><body><h1>🎬 VisEDA — Video EDA Report</h1><p><b>{inv.get("total_videos")}</b> videos · <b>{inv.get("valid_videos")}</b> valid · <b>{inv.get("corrupt_videos")}</b> corrupt</p><h2>📦 Inventory</h2><div class="grid"><div class="card"><h3>Counts</h3><div class="stat"><span>Total videos</span><span class="val">{inv.get("total_videos")}</span></div><div class="stat"><span>Valid videos</span><span class="val">{inv.get("valid_videos")}</span></div><div class="stat"><span>Corrupt videos</span><span class="val">{inv.get("corrupt_videos")}</span></div></div>{_bar("Label Distribution",inv.get("label_distribution"))}{_bar("Format Distribution",inv.get("format_distribution"))}</div><h2>🖼️ Spatial</h2><div class="grid">{_card("Height",sp.get("height"))}{_card("Width",sp.get("width"))}{_card("Aspect Ratio",sp.get("aspect_ratio"))}</div><h2>⏱️ Temporal Frame Statistics</h2><div class="grid">{_card("Frame Count",tm.get("frame_count"))}{_card("FPS",tm.get("fps"))}{_card("Duration",tm.get("duration_sec"))}{_card("Temporal Brightness Std",tm.get("temporal_brightness_std"))}</div><h2>🔎 Quality and Motion Blur</h2><div class="grid">{_card("Brightness",q.get("brightness_mean"))}{_card("Contrast",q.get("contrast_mean"))}{_card("Sharpness",q.get("sharpness_mean"))}{_card("Blur Fraction",q.get("blur_fraction"))}{_card("Motion Blur Fraction",mo.get("motion_blur_fraction"))}</div><h2>〰️ Motion</h2><div class="grid">{_card("Frame Difference",mo.get("frame_diff_mean"))}{_card("Motion Intensity",mo.get("motion_intensity_mean"))}{_card("Scene Change Rate",mo.get("scene_change_rate"))}</div><footer>Generated by VisEDA</footer></body></html>'
    Path(output_path).write_text(html_doc,encoding='utf-8')

__all__=['VideoEDA','VideoRecord']
