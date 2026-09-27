# Running the agent on a real SO-101

This is the setup we used on 26–27 September 2026. Several values are specific to our rig and are marked as such.
Keep a hand near the stop key whenever the arm is powered.

## Hardware

- An SO-101 follower arm (ours is a Hiwonder SO-ARM101) with its gripper camera.
- A top camera looking down at the table from above and in front of the arm. We used an iPhone as a Continuity Camera at
  1920×1440, rotated and cropped to a square. Turn off Center Stage: it reframes the image when it sees a person.
- A side camera (a USB webcam at 1280×720; its image is mirrored to match the simulator's side view).
- A Mac for the camera server (it uses AVFoundation) and a GPU server for the model (see the main README).

## 1. Arm server

The arm server is the only process that talks to the servos. It needs an SO-101 follower driver that provides
`so101_twin.hardware.CommandFollower` and `resolve_ports`; that driver is not part of this repository. Point the server at it
and at the follower's LeRobot-style calibration file:

```sh
export SO101_TWIN_DIR=/path/to/so101          # contains twin/ and resolve_ports.py
export SO101_CALIBRATION_JSON=/path/to/so101/twin/config.json
export SO101_PYTHON=/path/to/the/driver/venv/bin/python
open real/arm_server_fast.command            # its own terminal; 60 deg/s; press Enter in that window to STOP
```

The server caps joint speeds, holds position if the agent goes silent, and stops on Enter.

## 2. Camera server

Run it in a terminal that has macOS camera permission, and identify the camera indices by their images (OpenCV's order
differs from ffmpeg's device list):

```sh
python -m real.cam_server 0,1,2,3 --size 3=1920x1440,1=1280x720
```

If a camera stops delivering frames, the server restarts itself; clients retry for up to 25 s.

## 3. Calibration

The agent needs the top camera's pose and the arm's joint zero offsets, fitted together.

1. Paint a small green mark on one fingertip. Ours is on the **moving** jaw; `--marker-body` must name the jaw it is on.
   Calibrate with the gripper closed, so an empty close holds the moving jaw against the fixed one.
2. Clear the table, then sweep the whole workspace (about 150 poses, about 16 minutes):
   ```sh
   python -m real.fullcal real/cal/full.json --save-frames real/runs/full
   python -m real.contact_sheet real/runs/full          # look at every detection
   ```
3. Touch the table at a grid of spots (a single camera cannot see depth well; the touches fix the heights):
   ```sh
   python -m real.height_probe real/runs/height
   ```
4. Fit everything together and write the calibration:
   ```sh
   python -m real.kin_cal real/cal/full.json --contacts real/runs/height/probe.json \
       --models kin3 --model kin3 --contact-mm 0.7 --write
   cp real/cal/full_kin.json real/cal/current_camera.json
   ```
   Use the printed `table_z_world` as `--table-z` in `real/run_task.sh` and `real/run_fast.sh`.
5. Validate on fresh data through the runtime code before trusting it: run a short `fullcal` sweep and a `height_probe` with
   the new calibration installed. With ours, 20 fresh photos were predicted within 1.8 px and six fresh touches read between
   -0.8 and +4.7 mm.

## 4. Running commands

```sh
export MODEL_BASE_URL=https://<your-server>
bash real/run_fast.sh "put the blue ball in the square container, then put the yellow ball in the square container"
```

Each run starts and ends at the parked pose and writes `real/runs/ep_<time>/` (three-camera video, per-step frames, the
agent's decisions) plus a line in `real/runs/episodes.jsonl`. `run_task.sh` is the slower, more conservative mode.

**Rig-specific settings to adjust:** the camera indices, rotation, crop and expected size in the run scripts; `--table-z`
from your calibration; `--carry-z` (ours is 0.18 m for 10 cm tall containers); `--object-z` (the ball's centre height); and in
`real/arm.py` the base-joint limits for your walls (`PAN_MIN_DEG`, `PAN_MAX_DEG`) and the wrist-roll limit.

## Safety behaviour built in

- Travel moves rise first and stay at 13 cm or more with an empty gripper and 15.5 cm or more when carrying.
- A ball is dropped only if the carry actually arrived over the target, and never released without a real descent otherwise.
- Every run ends by rising and folding to the parked pose, including failures.
- Joint limits, a joint-jump guard, and a no-go zone near the wall are checked before every motion.
