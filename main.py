import cv2
import numpy as np

#url = "http://10.139.208.158:4747/video"
url = "http://10.7.135.103:4747/video"

cap = cv2.VideoCapture(url)
# cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def get_leftmost_point(contour):
    return contour[contour[:, :, 0].argmin()][0]

if not cap.isOpened():
    print("Error al abrir el video stream o archivo")
    exit()
    
commands = []

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("No se pudo obtener frame")
        break
    
    x, y, w, h = 40, 180, 560, 120

    box = frame[y:y+h, x:x+w]
    box_gray = cv2.cvtColor(box, cv2.COLOR_BGRA2GRAY)
    _, box_binary = cv2.threshold(box_gray, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)
    box_canny = cv2.Canny(box_binary, 100, 200)
    contours, _ = cv2.findContours(box_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_commands = []
    
    for contour in contours:
        contour += (x, y)
        area = cv2.contourArea(contour)
        epsilon = 0.09 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and area > 800:
            contours_commands.append(contour)
            
    cv2.drawContours(frame, contours_commands, -1, (0, 0, 255), 2)
    
    cv2.imshow("Box", box)
    
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    if len(contours_commands) > 0:
        contours_commands = sorted(contours_commands, key=lambda contour: get_leftmost_point(contour)[0])
        """
        leftmost_contour = contours_commands[0]
        rightmost_contour = contours_commands[0]
        
        for contour in contours_commands:
            leftmost_point = tuple(contour[contour[:,:,0].argmin()][0])
            rightmost_point = tuple(contour[contour[:,:,0].argmax()][0])
            
            # Comparar y actualizar el contorno más a la izquierda
            if leftmost_point[0] < tuple(leftmost_contour[leftmost_contour[:,:,0].argmin()][0])[0]:
                leftmost_contour = contour
                
            # Comparar y actualizar el contorno más a la derecha
            if rightmost_point[0] > tuple(rightmost_contour[rightmost_contour[:,:,0].argmax()][0])[0]:
                rightmost_contour = contour
                
        cv2.drawContours(frame, [leftmost_contour], -1, (255, 0, 0), 2)
        cv2.drawContours(frame, [rightmost_contour], -1, (0, 255, 0), 2)
        
        xp, yp, wp, hp = cv2.boundingRect(leftmost_contour)
        
        
        left = box_binary[yp-y:yp-y+hp, xp-x:xp-x+wp]
        
        cv2.imshow("Left", left)
        """
        
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        for contour in contours_commands:
            xp, yp, wp, hp = cv2.boundingRect(contour)
            print(xp - x, yp - y, wp, hp)
            
            contour_a = box_binary[yp-y+15:yp-y-15+hp, xp-x+15:xp-x-15+wp]
            
            contours_c = cv2.Canny(contour_a, 100, 200)
            contours_b, _ = cv2.findContours(contours_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            xb, yb, wb, hb = cv2.boundingRect(np.vstack(contours_b))
            ca = contour_a[yb:yb+hb, xb:xb+wb]
        # cv2.drawContours(frame, [c_b], -1, (255, 0, 0), 1)
            #print(xb, yb, wb, hb)
    
            mitad_h = (wb) // 2
            mitad_v = (hb) // 2
            
            izquierda = ca[:, :mitad_h]
            derecha = ca[:, mitad_h:]
            arriba = ca[:mitad_v, :]
            abajo = ca[mitad_v:, :]
            
            cv2.imshow(f"ca{mitad_h}{mitad_v}", ca)
            
            pixeles_negros_izquierda = cv2.countNonZero(izquierda)
            pixeles_negros_derecha = cv2.countNonZero(derecha)
            pixeles_negros_arriba = cv2.countNonZero(arriba)
            pixeles_negros_abajo = cv2.countNonZero(abajo)
            
            #cv2.imshow(f"{pixeles_negros_arriba}-{pixeles_negros_abajo}", arriba)
            #cv2.imshow(f"{pixeles_negros_abajo}-{pixeles_negros_arriba}", abajo)
            
            print(f"izquierda: {pixeles_negros_izquierda} - derecha: {pixeles_negros_derecha} - arriba: {pixeles_negros_arriba} - abajo: {pixeles_negros_abajo}")
            
            #print(max(pixeles_negros_arriba, pixeles_negros_abajo))
            #print(min(pixeles_negros_arriba, pixeles_negros_abajo))
            
            if pixeles_negros_izquierda < 50 and pixeles_negros_derecha < 50 and pixeles_negros_arriba < 50 and pixeles_negros_abajo < 50:
                commands.append(0)
                print("Es final\n")
            elif max(pixeles_negros_arriba, pixeles_negros_abajo) - min(pixeles_negros_arriba, pixeles_negros_abajo) > max(pixeles_negros_izquierda, pixeles_negros_derecha) - min(pixeles_negros_izquierda, pixeles_negros_derecha):
                if pixeles_negros_arriba > pixeles_negros_abajo:
                    commands.append(1)
                    print("La flecha apunta hacia la arriba\n")
                else:
                    commands.append(2)
                    print("La flecha apunta hacia abajo\n")
            elif max(pixeles_negros_izquierda, pixeles_negros_derecha) - min(pixeles_negros_izquierda, pixeles_negros_derecha) > max(pixeles_negros_arriba, pixeles_negros_abajo) - min(pixeles_negros_arriba, pixeles_negros_abajo):
                if pixeles_negros_izquierda > pixeles_negros_derecha:
                    commands.append(3)
                    print("La flecha apunta hacia la izquierda\n")
                else:
                    commands.append(4)
                    print("La flecha apunta hacia la derecha\n")
            
        break
    
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
for command in commands:
    if command == 0:
        continue
    elif command == 1:
        print("mover_adelante(1)")
    elif command == 2:
        print("girar_izquierda(2)")
        print("mover_adelante(1)")
        print("girar_derecha(2)")
    elif command == 3:
        print("girar_izquierda(1)")
    elif command == 4:
        print("girar_derecha(1)")

cap.release()
cv2.destroyAllWindows()
