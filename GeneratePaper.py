import cv2
import numpy as np

BG_COLOR = 209
BG_SIGMA = 5
MONOCHROME = 1
class GeneratePaper():
    def __init__(self) -> None:
        self.BG_COLOR = 209
        self.BG_SIGMA = 5
        self.MONOCHROME = 1

    def blank_image(self, width=1024, height=1024, background=BG_COLOR):
        """
        It creates a blank image of the given background color
        """
        img = np.full((height, width, self.MONOCHROME), background, np.uint8)
        # print("blank_image ", width, height)
        # input()
        return img
    def add_noise(self, img, sigma=BG_SIGMA):
        """
        Adds noise to the existing image
        """
        width, height, ch = img.shape
        # print("add_noise: ", width, height)
        # input()
        n = self.noise(width, height, sigma=sigma)
        img = img + n
        return img.clip(0, 255)


    def noise(self, width, height, ratio=1, sigma=BG_SIGMA):
        """
        The function generates an image, filled with gaussian nose. If ratio parameter is specified,
        noise will be generated for a lesser image and then it will be upscaled to the original size.
        In that case noise will generate larger square patterns. To avoid multiple lines, the upscale
        uses interpolation.

        :param ratio: the size of generated noise "pixels"
        :param sigma: defines bounds of noise fluctuations
        """
        mean = 0
        # print("noise: ", width, height, ratio)
        # input()
        
        # assert width % ratio == 0, "Can't scale image with of size {} and ratio {}".format(width, ratio)
        # assert height % ratio == 0, "Can't scale image with of size {} and ratio {}".format(height, ratio)

        h = int(height / ratio)
        w = int(width / ratio)

        result = np.random.normal(mean, sigma, (w, h, self.MONOCHROME))
        if ratio > 1:
            result = cv2.resize(result, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        return result.reshape((width, height, self.MONOCHROME))


    def texture(self, image, sigma=BG_SIGMA, turbulence=2):
        """
        Consequently applies noise patterns to the original image from big to small.

        sigma: defines bounds of noise fluctuations
        turbulence: defines how quickly big patterns will be replaced with the small ones. The lower
        value - the more iterations will be performed during texture generation.
        """
        result = image.astype(float)
        cols, rows, ch = image.shape
        ratio = cols
        while not ratio == 1:
            if cols % ratio != 0 and rows % ratio != 0:
                cur_noise = self.noise(cols, rows, ratio, sigma=sigma)
                result += cur_noise
            ratio = (ratio // turbulence) or 1
        cut = np.clip(result, 0, 255)
        cut = cut.astype(np.uint8)
        cut -= 50
        return cut.astype(np.uint8)
if __name__=="_main__":
    bg_paper=GeneratePaper()
    # cv2.imwrite('texture.jpg', bg_paper.texture(bg_paper.blank_image(background=230), sigma=4, turbulence=4))
    img=bg_paper.blank_image(width=120, height=120, background=230)
    img=bg_paper.texture(img, sigma=4)
    img=bg_paper.add_noise(img, sigma=10)
    # cv2.imwrite('noise.jpg', bg_paper.add_noise(bg_paper.blank_image(1024, 1024), sigma=10))