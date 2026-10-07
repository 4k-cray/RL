import gymnasium as gym
import numpy as np

class CaptureVideoWrapper(gym.Wrapper):
    def __init__(self, env, writer=None):
        super(CaptureVideoWrapper, self).__init__(env)
        self.capture_mode = False
        self.frames = []
        self.video_frames = []
        self.writer = writer

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.video_frames += self.frames
        self.frames = []  # Clear previous frames at the start of a new episode
        return observation

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        
        if self.capture_mode:
            frame = self.env.render()
            # Capture the current frame
            if frame is not None:
                self.frames.append(frame)
        
        return observation, reward, done, truncated, info

    def save_frames_to_tensorboard(self, global_step, success = None, fps=30):
        if not self.frames:
            return
        
        # Add the "Episode {global_step}" text to all frames
        from PIL import Image, ImageDraw, ImageFont
        import os

        success_font = ImageFont.truetype(f"{os.path.dirname(os.path.abspath(__file__))}/font_success.ttf", 60)
        fail_font = ImageFont.truetype(f"{os.path.dirname(os.path.abspath(__file__))}/font_fail.ttf", 60)

        for i, frame in enumerate(self.frames):
            pil_image = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_image)
            draw.text((10, 10), f"Episode {global_step}", fill=(255, 0, 0))

            # add success/fail text in the last 10% of the frames
            if i >= len(self.frames) - fps:
                if success is not None:
                    color = (0, 255, 0) if success else (255, 0, 0)
                    text = "Success!" if success else "FAIL"
                    font = success_font if success else fail_font

                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    frame_width, frame_height = pil_image.size
                    position = ((frame_width - text_width) // 2, (frame_height - text_height) // 2)
                
                    draw.text(position, text, fill=color, font=font)
            
            frame_with_text = np.array(pil_image)
            self.frames[i] = frame_with_text
        
        # Convert frames to the appropriate format for TensorBoard
        video = np.array(self.frames)  # Shape: (time, height, width, channels)
        video = video.transpose(0, 3, 1, 2)  # Convert to (time, channels, height, width)
        video = video[np.newaxis, ...]  # Add batch dimension, now (1, time, channels, height, width)
        
        # Log the video to TensorBoard
        if self.writer:
            self.writer.add_video(f'Agent', video, fps=fps, global_step=global_step)

        self.capture_mode = False

    def save_to_video(self, filename: str, fps: int = 30):
        if not self.video_frames:
            print("No frames captured. Ensure capture_mode is on.")
            return
        
        # Save the frames to an mp4 video file
        from moviepy.editor import ImageSequenceClip

        # Create an ImageSequenceClip from the list of frames
        clip = ImageSequenceClip(self.video_frames, fps=fps)
        
        # Write the clip to a file
        clip.write_videofile(filename, codec='libx264', logger=None)

    def activate(self):
        self.capture_mode = True