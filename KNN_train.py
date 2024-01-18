import sys
import numpy as np
import cv2
import os

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

def main():
    pic="22"
    imgTrainNos = cv2.imread("uploads/"+pic+".jpg")
    #imgTrainNos = cv2.imread("C:/Users/USER/source/repos/detect_car/detect_car/result/oneline/oneline"+pic+".jpg")

    if imgTrainNos is None:
        print("error: image not read from the given file! \n\n")
        os.system("pause")
        return
    #end


    imageGray = cv2.cvtColor(imgTrainNos, cv2.COLOR_BGR2GRAY)
    imageBlurred = cv2.GaussianBlur(imageGray, (5,5), 0)

    imageThresh = cv2.adaptiveThreshold(imageBlurred,
                                        255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV,
                                        17,
                                        2)

    cv2.imshow("imageThresh", imageThresh)

    imageThreshCopy = imageThresh.copy()

    imageContours, npaContours, npaHierarchy = cv2.findContours(imageThreshCopy,
                                                                cv2.RETR_EXTERNAL,
                                                                cv2.CHAIN_APPROX_SIMPLE)

    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []

    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('a'), ord('b'), ord('c'), ord('d'), ord('e'), ord('f'), ord('g'), ord('h'), ord('i'), ord('j'),
                     ord('k'), ord('l'), ord('m'), ord('n'), ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'),ord('u'),ord('v'),ord('w'),ord('x'),ord('y'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'),ord('G'),ord('H'),ord('I'),ord('J'),ord('K'),ord('L'),ord('M'),ord('N'),ord('z')]

    for npaContour in npaContours:
        if cv2.contourArea(npaContour)> MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            cv2.rectangle(imgTrainNos,
                         (intX, intY),
                         (intX+intW,intY+intH),
                         (0,0,255),
                          2)


            imageCrop = imageThresh[intY:intY+intH, intX:intX+intW]
            imageCropResized = cv2.resize(imageCrop, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))


            cv2.imshow("imageCrop", imageCrop)
            cv2.imshow("imageCropResized", imageCropResized)
            cv2.imshow("training_numbers.png",imgTrainNos)


            intChar = cv2.waitKey(0)

            if intChar == 27:
               sys.exit()
##################   kanji ##############################
            elif intChar in intValidChars:
                if intChar == ord('a'):
                    intChar = ord('横')

                if intChar == ord('b'):
                    intChar = ord('浜')

                if intChar == ord('c'):
                    intChar = ord('練')

                if intChar == ord('d'):
                    intChar = ord('馬')

                if intChar == ord('e'):
                    intChar = ord('松')

                if intChar == ord('f'):
                    intChar = ord('豊')

                if intChar == ord('g'):
                    intChar = ord('田')

                if intChar == ord('h'):
                    intChar = ord('品')  
                    
                if intChar == ord('i'):
                    intChar = ord('川')

                if intChar == ord('j'):
                    intChar = ord('湘')

                if intChar == ord('k'):
                    intChar = ord('南')

                if intChar == ord('l'):
                    intChar = ord('足')

                if intChar == ord('m'):
                    intChar = ord('立')

                if intChar == ord('n'):
                    intChar = ord('神')

                if intChar == ord('o'):
                    intChar = ord('戸')
                    
                if intChar == ord('p'):
                    intChar = ord('宇')

                if intChar == ord('q'):
                    intChar = ord('都')

                if intChar == ord('r'):
                    intChar = ord('宮')

                if intChar == ord('s'):
                    intChar = ord('日')

                if intChar == ord('t'):
                    intChar = ord('野')

                if intChar == ord('u'):
                    intChar = ord('名')

                if intChar == ord('v'):
                    intChar = ord('古')

                if intChar == ord('w'):
                    intChar = ord('群')

                if intChar == ord('x'):
                    intChar = ord('屋')

#######################  hiragana  ####################
                if intChar == ord('A'):
                    intChar = ord('む')

                if intChar == ord('B'):
                    intChar = ord('ひ')

                if intChar == ord('C'):
                    intChar = ord('な')

                if intChar == ord('D'):
                    intChar = ord('ぬ')

                if intChar == ord('E'):
                    intChar = ord('ふ')

                if intChar == ord('F'):
                    intChar = ord('て')

                if intChar == ord('G'):
                    intChar = ord('ね')

                if intChar == ord('H'):
                    intChar = ord('る')

                if intChar == ord('I'):
                    intChar = ord('は')

                if intChar == ord('J'):
                    intChar = ord('つ')

                if intChar == ord('K'):
                    intChar = ord('ざ')

                if intChar == ord('L'):
                    intChar = ord('ら')

                if intChar == ord('M'):
                    intChar = ord('た')

                if intChar == ord('N'):
                    intChar = ord('う')


                
                if intChar == ord('z'):
                    intChar = ord(' ')

############################################################


                intClassifications.append(intChar)

                npaFlattenedImage = imageCropResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)

            #endif
        #endif
    #endfor


    floatClassifications = np.array(intClassifications, np.float32)
    npaClassifications= floatClassifications.reshape(floatClassifications.size, 1)

    print("\n TRAINING DONE!\n")

    np.savetxt("trianingFile/Classifications"+pic+".txt", npaClassifications)
    np.savetxt("trianingFile/FlattenedImages"+pic+".txt", npaFlattenedImages)

    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()
# end if
     
            
        
    

