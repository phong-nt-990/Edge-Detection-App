from customtkinter import CTk, set_appearance_mode, CTkFrame, CTkLabel, CTkButton, CTkImage, StringVar, CTkRadioButton
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter as tk
import cv2
from scipy.signal import convolve2d

class App(CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1200x600")
        self.resizable(0, 0)
        self.title("Edge Detection App")
        set_appearance_mode("light")

        self.original_image = None
        self.processed_image = None

        self.main_view = CTkFrame(master=self, fg_color="#ffffff", width=1350, height=600, corner_radius=0)
        self.main_view.pack_propagate(0)
        self.main_view.pack()


        self.sidebar_frame = CTkFrame(master=self.main_view, fg_color="#fff", width=190, height=650, corner_radius=0)
        self.sidebar_frame.pack_propagate(0)
        self.sidebar_frame.pack(fill="y", anchor="w", side="left")
        


        CTkLabel(master=self.sidebar_frame, text="Edge Detection App", font=("Arial Black", 14)).pack(pady=(38, 0), anchor="center")

        self.load_image_button = CTkButton(master=self.sidebar_frame, text="Load Image", text_color="#fff", command=self.load_image, font=("Arial Black", 12))
        self.load_image_button.pack(anchor="center", ipady=5, pady=(30, 0))

        self.check_button = CTkButton(master=self.sidebar_frame, text="Start detection", font=("Arial Black", 12), command=self.process_image,
                                      text_color="#fff")
        self.check_button.pack(anchor="center", ipady=5, pady=(30, 0))

        self.refresh_button = CTkButton(master=self.sidebar_frame, text="Refresh", font=("Arial Black", 12), command=self.refresh_button_signal, 
                                      text_color="#fff")
        self.refresh_button.pack(anchor="center", ipady=5, pady=(30, 0))

        CTkLabel(master=self.sidebar_frame, text="Select mode", font=("Arial Black", 12)).pack(pady=(38, 0), anchor="center")

        self.edge_detection_mode = StringVar(value="Prewitt")

        self.prewitt_radio_button = CTkRadioButton(master=self.sidebar_frame, text="Prewitt operator", variable=self.edge_detection_mode, value="Prewitt")
        self.prewitt_radio_button.pack(anchor="w", pady=(10, 0), padx=(25,0))

        self.sobel_radio_button = CTkRadioButton(master=self.sidebar_frame, text="Sobel operator", variable=self.edge_detection_mode, value="Sobel")
        self.sobel_radio_button.pack(anchor="w", pady=(10, 0), padx=(25,0))

        self.freichen_radio_button = CTkRadioButton(master=self.sidebar_frame, text="Frei-chen operator", variable=self.edge_detection_mode, value="FreiChen")
        self.freichen_radio_button.pack(anchor="w", pady=(10, 0), padx=(25,0))

        self.laplacian_radio_button = CTkRadioButton(master=self.sidebar_frame, text="Laplacian operator", variable=self.edge_detection_mode, value="Laplacian")
        self.laplacian_radio_button.pack(anchor="w", pady=(10, 0), padx=(25,0))

        self.canny_radio_button = CTkRadioButton(master=self.sidebar_frame, text="Canny method", variable=self.edge_detection_mode, value="Canny")
        self.canny_radio_button.pack(anchor="w", pady=(10, 0), padx=(25,0))


        self.left_frame = CTkFrame(master=self.main_view, fg_color="#fff", width=490, height=600, corner_radius=0)
        self.left_frame.pack_propagate(0)
        self.left_frame.pack(side='left')

        self.right_frame = CTkFrame(master=self.main_view, fg_color="#fff", width=490, height=600, corner_radius=0)
        self.right_frame.pack_propagate(0)
        self.right_frame.pack(side='right')


        self.left_image_frame = CTkFrame(master=self.left_frame, fg_color="#dddddd", width=555, height=500, corner_radius=0)
        self.left_image_frame.pack_propagate(0)
        self.left_image_frame.pack(side='left', padx=(5, 0))

        self.left_image_label = CTkLabel(master=self.left_image_frame, text="Please input image.", font=("Arial Black", 20), text_color="#37383a")
        self.left_image_label.pack(anchor="center", pady=(225, 0))

        self.right_image_frame = CTkFrame(master=self.right_frame, fg_color="#dddddd", width=555, height=500, corner_radius=0)
        self.right_image_frame.pack_propagate(0)
        self.right_image_frame.pack(side='right', padx=(0,25))

        self.right_image_label = CTkLabel(master=self.right_image_frame, text="Result image", font=("Arial Black", 20), text_color="#37383a")
        self.right_image_label.pack(anchor="center", pady=(225, 0))
        # self.left_text_label = CTkLabel(self.left_image_frame, text="Input image to here", font=('Arial', 12))
        # self.left_text_label.pack(pady=10)

        # self.right_text_label = CTkLabel(self.right_image_frame, text="Output image after processing", font=('Arial', 12))
        # self.right_text_label.pack(pady=10)

        self.input_image = CTkLabel(self.left_image_frame, text="")
        self.output_image = CTkLabel(self.right_image_frame, text="")
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])  # Opens a file dialog window for selecting an image
        if file_path:  # If a file is selected
            self.original_image = Image.open(file_path)
            self.original_image.thumbnail((485, 500))  # Resize the image to fit in the frame

            # Display the image in the left frame
            photo = ImageTk.PhotoImage(self.original_image)
            self.left_image_label.pack_forget()
            self.input_image.configure(image=photo)       

            height = self.left_image_frame.winfo_height()

            y = int((height - self.original_image.height) // 2)
        

            self.input_image.pack(padx=(0, 0), pady=(y,0))
    # def load_image(self):
    #     file_path = "test.jpg"  # Replace this with your image file path
    #     self.original_image = Image.open(file_path)
    #     self.original_image.thumbnail((550, 500))  # Resize the image to fit in the frame

    #     # Display the image in the left frame
    #     photo = ImageTk.PhotoImage(self.original_image)
    #     self.left_image_label.pack_forget()
    #     self.input_image.configure(image=photo)       
    #     self.input_image.pack()


    def process_image(self):
        if self.original_image is not None:
            # Convert PIL image to OpenCV format (numpy array)
            cv_image = cv2.cvtColor(cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)

            # Increase brightness by adding a value of 1 to all pixels
            if self.edge_detection_mode.get() == "Sobel":
                processed_image = self.SobelOperatorDetection(cv_image)
            elif self.edge_detection_mode.get() == "Prewitt":
                processed_image = self.PrewittOperatorDetection(cv_image)
            elif self.edge_detection_mode.get() == "Canny":
                processed_image = self.CannyEdgeDetection(cv_image)
            elif self.edge_detection_mode.get() == "Laplacian":
                processed_image = self.LaplacianOperatorEdgeDetection(cv_image)
            elif self.edge_detection_mode.get() == "FreiChen":
                processed_image = self.FreiChenOperatorEdgeDetection(cv_image)
            else:
                raise "ERROR"
            # Convert back to PIL format for display
            self.processed_image = Image.fromarray(processed_image)
            self.processed_image.thumbnail((550, 500))  # Resize processed image to fit in the frame

            self.right_image_label.pack_forget()
            # Display processed image in the right frame
            processed_photo = ImageTk.PhotoImage(self.processed_image)
            self.output_image.configure(image=processed_photo)

            height = self.right_image_frame.winfo_height()

            y = int((height - self.processed_image.height) // 2)
        
            self.output_image.pack(padx=(0,0), pady=(y, 0))
        else:
            print("Please load an image first.")

    def refresh_button_signal(self):
        if self.original_image is not None:
            # Clear images and reset labels
            self.left_image_label.pack_forget()
            self.right_image_label.pack_forget()
            self.input_image.configure(image=None)
            self.output_image.configure(image=None)
            self.input_image.pack_forget()
            self.output_image.pack_forget()

            # Re-show default text in labels
            self.left_image_label.configure(text="Please input image.")
            self.right_image_label.configure(text="Result image")
            self.left_image_label.pack(anchor="center", pady=(225, 0))
            self.right_image_label.pack(anchor="center", pady=(225, 0))
            self.original_image = None
        else:
            print("Please load an image first.")


    def PrewittOperatorDetection(self, img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Khởi tạo 2 kernel Prewitt theo x-dim và y-dim
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
        img_gaussian = cv2.GaussianBlur(gray_image,(3,3),0)
        # Convolution lần lượt 2 kernel vào ảnh
        img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
        # Tính tổng
        img_prewitt = img_prewittx + img_prewitty
        return img_prewitt

    def FreiChenOperatorEdgeDetection(self, img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gaussian = cv2.GaussianBlur(gray_image,(3,3),0)
        SQRT2 = np.sqrt(2)
        frei_chen_x = np.array([[-1, 0, 1], [-SQRT2, 0, SQRT2], [-1, 0, 1]])
        frei_chen_y = np.array([[1, SQRT2, 1], [0, 0, 0], [-1, -SQRT2, -1]])

        # Convolution lần lượt 2 kernel vào ảnh
        g_x = convolve2d(img_gaussian, frei_chen_x, mode='same', boundary='symm') 
        g_y = convolve2d(img_gaussian, frei_chen_y, mode='same', boundary='symm') 

        # Tính độ lớn của gradient
        magnitude = np.hypot(g_x, g_y)

        # Chuẩn hoá độ lớn
        magnitude *= 255.0 / np.max(magnitude)

        # Threshold
        edge_map = magnitude > 30 
        return edge_map

    def CannyEdgeDetection(self, img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gaussian = cv2.GaussianBlur(gray_image,(3,3),0)

        # Define Sobel operator
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        # Tính toán cường độ và hướng của Gradient, sử dụng toán tử Sobel
        g_x = convolve2d(img_gaussian, sobel_x, mode='same', boundary='symm') # img_gray đã được lọc từ img_gaussian từ trước
        g_y = convolve2d(img_gaussian, sobel_y, mode='same', boundary='symm') 

        # Tính toán độ lớn và hướng của gradient
        magnitude = np.hypot(g_x, g_y)
        angle = np.arctan2(g_y, g_x) * 180. / np.pi
        angle[angle < 0] += 180

        # Đặt giá trị cho 2 ngưỡng
        high_threshold = 120
        low_threshold = 200

        M, N = gray_image.shape
        res = np.zeros((M,N), dtype=np.int32)

        # Đặt giá trị cho độ lớn pixel cạnh yếu và cạnh mạnh
        weak = np.int32(25)
        strong = np.int32(255)


        # Áp dụng ngưỡng kép
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (magnitude[i,j] >= high_threshold):
                    res[i,j] = strong
                elif (low_threshold <= magnitude[i,j] < high_threshold):
                    if ((magnitude[i-1, j-1:j+2] >= high_threshold).any() or
                        (magnitude[i+1, j-1:j+2] >= high_threshold).any() or
                        (magnitude[i, [j-1, j+1]] >= high_threshold).any()):
                        res[i, j] = strong
                    else:
                        res[i, j] = weak

        # Kiểm tra và chuyển các độ lớn pixel cạnh yếu thành pixel cạnh mạnh
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (res[i,j] == weak):
                    try:
                        if ((res[i-1, j-1:j+2] == strong).any() or
                            (res[i+1, j-1:j+2] == strong).any() or
                            (res[i, [j-1, j+1]] == strong).any()):
                            res[i, j] = strong
                        else:
                            res[i, j] = 0
                    except IndexError as e:
                        pass

        # Lọc và giữ lại các pixel cạnh mạnh
        edge_map = res == strong
        return edge_map

    def SobelOperatorDetection(self, img):
        # doing sth
        # Khởi tạo 2 kernel Sobel theo x-dim và y-dim
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gaussian = cv2.GaussianBlur(gray_image,(3,3),0)
        kernelx = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        kernely = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        # Convolution lần lượt 2 kernel vào ảnh
        img_sobelx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_sobely = cv2.filter2D(img_gaussian, -1, kernely)
        # Tính tổng
        img_sobel = img_sobelx + img_sobely
        return img_sobel

    def LaplacianOperatorEdgeDetection(self,img):
        
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gaussian = cv2.GaussianBlur(gray_image,(3,3),0)

        # Khởi tạo kernel Laplacian
        kernel = np.array([[-1, -1, -1], 
                            [-1, 8, -1], 
                            [-1, -1, -1]]) 

        # Convolution kernel vào ảnh
        img_laplacian = convolve2d(img_gaussian, kernel, mode='same', boundary='symm') 


        # Tìm điểm zero crossing
        zero_crossings = np.zeros_like(img_laplacian, dtype=np.uint8)
        height, width = gray_image.shape
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                patch = img_laplacian[i-1:i+2, j-1:j+2]
                if np.any(patch > 0) and np.any(patch < 0):
                    zero_crossings[i, j] = 255
        return zero_crossings

def main():
    app = App()
    app.mainloop()

if __name__ == '__main__':
    main()
