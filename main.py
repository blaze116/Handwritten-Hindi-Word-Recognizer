from seg_utils import *
from model_utils import *

def predict(image):
    loc = Localizer(image)
    model = tf.keras.models.load_model("model.h5")
    crop_img = loc.get_crop()
    seg = Segmenter(crop_img)
    answer = []
    for crop in seg.segment():
            # crop1 = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        crop = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 7)

        crop = pad_image(crop)
        crop = cv2.resize(crop, (32, 32)).reshape(1, 32, 32, 1) / 255.0
        pred = np.squeeze(model.predict(crop))
        pred = np.argmax(pred)
        answer.append(mapping[pred+1])
    return answer


def test():
    '''
    We will be using a similar template to test your code
    '''
    image_paths = ["images/sl_1.jpg", "images/sl_2.jpg", "images/sl_3.jpg", "images/sl_4.jpg"]
    correct_answers = [["र", "व", "न"], ["त", "ल", "प"], ["स", "ह", "त्र"], ["ठ", "त", "थ", "छ", "च"]]
    score = 0
    multiplication_factor=2 #depends on character set size

    for i,image_path in enumerate(image_paths):
        image = cv2.imread(image_path) # This input format wont change
        answer = predict(image) # a list is expected
        print(''.join(answer))# will be the output string

        n=0
        for j in range(len(answer)):
            if correct_answers[i][j] == answer[j]:
                n+=1
                
        if(n==len(correct_answers[i])):
            score += len(correct_answers[i])*multiplication_factor
        else:
            score += n*2
        
    
    print('The final score of the participant is',score)


if __name__ == "__main__":
    test()