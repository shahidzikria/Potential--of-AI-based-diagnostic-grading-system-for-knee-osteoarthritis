from src.preprocessing import *
from src.modeling import *

# Example usage:
if __name__ == "__main__":
    # 1. Load and preprocess data
    df = pd.read_csv("knee_xray.csv")
    X, y = [], []

    for k in range(len(df)):
        path = f"./dataset/{df['SERIAL NO.'][k]}.jpg"
        img = read_img(path)
        img = fix_pixels(img)
        img = sharpen_edges(img)
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        img = img.T
        points = detect_knee_points(opened.T, threshold=60)
        sp = clean_points(points)
        left, right = img[sp[0]:sp[1]].T, img[sp[2]:sp[3]].T
        X.extend([left, right])
        y.extend([df["LEFT KNEE"][k], df["RIGHT KNEE"][k]])

    X = np.array(X)
    y = np.array(y)
    y = one_hot_encode(y, 5)

    # 2. Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y)

    # 3. Model creation and training
    model = build_model()
    model = compile_model(model)
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=50)

    # 4. Save model
    model.save("knee_classifier.h5")
