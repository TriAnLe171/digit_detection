import cv2

def increase_contrast(image):
    if len(image.shape) == 2 :
        normalized = cv2.normalize(image,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
        equalized=cv2.equalizeHist(normalized)

        return equalized
    elif len(image.shape)==3 and image.shape[2]==3:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        normalized=cv2.normalize(gray,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
        equalized=cv2.equalizeHist(normalized)
        contrast=cv2.cvtColor(equalized,cv2.COLOR_GRAY2BGR)
        return contrast
    else:
        ValueError('NNO')
