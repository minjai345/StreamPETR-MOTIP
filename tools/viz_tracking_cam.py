"""Visualize tracking results on 6 camera views as GIF per scene.

Projects 3D tracking boxes onto each camera image. Each tracking_id
gets a consistent color across frames. Outputs one GIF per scene.

Usage:
    PYTHONPATH=$(pwd):$PYTHONPATH python tools/viz_tracking_cam.py \
        --submission work_dirs/motip_phase1_v1/tracking_results.json \
        --scene scene-0003 \
        --out work_dirs/motip_phase1_v1/viz/
"""
import os, sys, json, argparse
import numpy as np
import cv2
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

CAM_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
]

# 20 distinct colors for tracking IDs
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (0, 128, 255), (128, 0, 255), (255, 128, 128), (128, 255, 128),
]


def get_color(tracking_id):
    return COLORS[hash(tracking_id) % len(COLORS)]


def draw_box_on_img(img, corners_2d, color, tracking_id, score):
    """Draw 3D box edges projected to 2D on image."""
    corners = corners_2d.astype(int)
    # Bottom face
    for i in range(4):
        j = (i + 1) % 4
        cv2.line(img, tuple(corners[:, i]), tuple(corners[:, j]), color, 2)
    # Top face
    for i in range(4, 8):
        j = 4 + (i - 4 + 1) % 4
        cv2.line(img, tuple(corners[:, i]), tuple(corners[:, j]), color, 2)
    # Pillars
    for i in range(4):
        cv2.line(img, tuple(corners[:, i]), tuple(corners[:, i + 4]), color, 1)
    # Label
    cx, cy = int(corners[0].mean()), int(corners[1].min()) - 5
    label = f'{tracking_id}'
    cv2.putText(img, label, (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def render_frame(nusc, sample_token, detections, dataroot):
    """Render one frame: 6 cameras side by side with projected boxes."""
    sample = nusc.get('sample', sample_token)
    imgs = []

    for cam in CAM_ORDER:
        sd_token = sample['data'][cam]
        sd = nusc.get('sample_data', sd_token)
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ep = nusc.get('ego_pose', sd['ego_pose_token'])

        img_path = os.path.join(dataroot, sd['filename'])
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((900, 1600, 3), dtype=np.uint8)
        h, w = img.shape[:2]

        # Camera intrinsic
        intrinsic = np.array(cs['camera_intrinsic'])

        # Transforms
        ego2global_r = Quaternion(ep['rotation'])
        ego2global_t = np.array(ep['translation'])
        sensor2ego_r = Quaternion(cs['rotation'])
        sensor2ego_t = np.array(cs['translation'])

        for det in detections:
            # Create Box in global frame
            box = Box(
                center=det['translation'],
                size=det['size'],
                orientation=Quaternion(det['rotation']),
                score=det['tracking_score'],
                name=det['tracking_name'],
                token=det['tracking_id'],
            )

            # Global → ego
            box.translate(-ego2global_t)
            box.rotate(ego2global_r.inverse)
            # Ego → sensor
            box.translate(-sensor2ego_t)
            box.rotate(sensor2ego_r.inverse)

            # Check if box is in front of camera
            if box.center[2] <= 0:
                continue

            # Project corners to image
            corners_3d = box.corners()  # [3, 8]
            corners_2d = view_points(corners_3d, intrinsic, normalize=True)[:2]  # [2, 8]

            # Check if any corner is in image
            in_img = (
                (corners_2d[0] >= 0) & (corners_2d[0] < w) &
                (corners_2d[1] >= 0) & (corners_2d[1] < h)
            )
            if not in_img.any():
                continue

            color = get_color(det['tracking_id'])
            draw_box_on_img(img, corners_2d, color, det['tracking_id'], det['tracking_score'])

        # Resize for GIF
        scale = 400 / h
        img = cv2.resize(img, (int(w * scale), 400))
        imgs.append(img)

    # Concat: top row [FL, F, FR], bottom row [BL, B, BR]
    top = np.concatenate(imgs[:3], axis=1)
    bottom = np.concatenate(imgs[3:], axis=1)
    frame = np.concatenate([top, bottom], axis=0)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', required=True)
    parser.add_argument('--scene', default='scene-0003')
    parser.add_argument('--out', default='viz/')
    parser.add_argument('--dataroot', default='data/nuscenes/')
    parser.add_argument('--score-thresh', type=float, default=0.15)
    parser.add_argument('--max-frames', type=int, default=40)
    args = parser.parse_args()

    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)

    with open(args.submission) as f:
        sub = json.load(f)

    # Find scene
    scene = None
    for s in nusc.scene:
        if s['name'] == args.scene:
            scene = s
            break
    assert scene is not None, f'Scene {args.scene} not found'

    # Get tokens
    tokens = []
    tok = scene['first_sample_token']
    while tok:
        tokens.append(tok)
        sample = nusc.get('sample', tok)
        tok = sample['next']
    tokens = tokens[:args.max_frames]

    print(f'Scene: {args.scene}, {len(tokens)} frames')

    # Render frames
    frames = []
    for i, tok in enumerate(tokens):
        dets = sub['results'].get(tok, [])
        dets = [d for d in dets if d['tracking_score'] >= args.score_thresh]
        frame = render_frame(nusc, tok, dets, args.dataroot)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        if (i + 1) % 10 == 0:
            print(f'  [{i+1}/{len(tokens)}] {len(dets)} dets')

    # Save GIF
    os.makedirs(args.out, exist_ok=True)
    gif_path = os.path.join(args.out, f'{args.scene}.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=500, loop=0)
    print(f'Saved: {gif_path}')


if __name__ == '__main__':
    main()
