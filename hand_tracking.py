import cv2
import mediapipe as mp


class HandDetector:
    """
    Hand detector class
    """

    def __init__(self, static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=0,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Creates an instance of HandDetector class
        """

        # Mediapipe's hand tracking class
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.static_image_mode,
                                         max_num_hands=self.max_num_hands,
                                         model_complexity=self.model_complexity,
                                         min_detection_confidence=self.min_detection_confidence,
                                         min_tracking_confidence=self.min_tracking_confidence)

        # Utility to draw points and lines over hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def find_hands(self, img, draw=True, return_landmarks=False):
        """
        Method to find hands, draw and/or landmarks
        :param img: Image or video capture
        :param draw: Whether to draw landmarks
        :param return_landmarks: Whether to return landmarks
        :return: original img or landmarks drawn image depending on "draw" parameter, landmarks' ids and
        coordinates or empty list depending on "return_landmarks" parameter
        """

        height, width, _ = img.shape

        # Flipping the capture around y-axis for correct handedness
        img = cv2.flip(img, 1)

        # Convert the BGR image to RGB before processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        landmarks = []

        if results.multi_hand_landmarks:
            # Drawings for each hand

            for hand in results.multi_hand_landmarks:
                # Creating a list of tuples including ids and coordinates of landmarks

                if return_landmarks:
                    hand_landmarks = [(id_landmark, int(landmark.x * width), int(landmark.y * height))
                                      for id_landmark, landmark in enumerate(hand.landmark)]
                    landmarks.append(hand_landmarks)
                if draw:
                    self.mp_drawing.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS,
                                                   self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                                   self.mp_drawing_styles.get_default_hand_connections_style())
                    if return_landmarks:
                        [cv2.circle(img, (cx, cy), 15, (255, 0, 255)) for _, cx, cy in hand_landmarks]

            # Return the img with drawings to show
        return img, landmarks
