import cv2
import os

cwd = os.getcwd()
plot_dir = cwd + '/plots/incremental_geocert/'
film_dir = cwd + '/plots/'
fps = 24

image_folder = plot_dir
video_name = film_dir + 'iterative_geocert.avi'

# images = [img for img in os.listdir(image_folder) if img.endswith(".svg")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape
#
# video = cv2.VideoWriter(video_name, 0, 1, (width,height))
#
# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))
#
# cv2.destroyAllWindows()
# video.release()



def save_film(plot_dir, fps, film_dir):
    string = "ffmpeg -r "+str(fps)+" -i "+plot_dir+"%01d.png -vcodec mpeg4 -y "+film_dir+"movie.mp4"
    os.system(string)
    print('movie saved')


save_film(plot_dir, fps, film_dir)