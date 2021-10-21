import moviepy.editor as mp
class VideoToAudioConvert:
    def videoConvert(self):
        clip = mp.VideoFileClip(r"Sound\Zarin_Video.mp4")
        clip.audio.write_audiofile(r"Zarin_Video.wav")  

