"""Pytest tests for the Flask app: routes, edge cases, preprocessing, prediction, everything"""

import io #lets us build a file in memory instead of disk

from PIL import Image #used to draw small test pictures

from model import preprocess_img, predict_result #the 2 functions we will be testing

GOOD_PICTURE = "test_images/4/Sign 4 (142).jpeg" #handsign already in our app
ERROR_TEXT = b"File cannot be processed." #message result.html shows on failure

def make_image(mode="RGB", size = (40, 40)):
    """Make small image in memory, so we do not need an extra file"""
    buf = io.BytesIO() #empty file in memory
    Image.new(mode, size).save(buf, format="PNG") #draw image and save it into it
    buf.seek(0) #go back to start so it can be read
    return buf #hand back the in memory image

def number_shown(page):
    """Returns true if page shows a single digit as a prediction"""
    return any (f">{digit}<".encode() in page for digit in range(10)) #looks for number 0 to 9

#Flask Routes, normal case
def test_home_page_load(client):
    """Normal case: the home page opens without issue and shows app name"""
    response = client.get("/") #Visit the home page
    assert response.status_code == 200 #200 = OK
    assert b"Hand Sign Digit Language Detection" in response.data #title is on page

def test_home_page_has_upload_form(client):
    """Normal case: the home page has a fil input that goes to /prediction"""
    response = client.get("/") #visit the home page
    assert b'name="file"' in response.data #the upload box is there
    assert b"/prediction" in response.data #the form sends to /prediction

def test_valid_image_gets_prediction(client):
    """A real hand sign image shows a sigit, not an error message"""
    with open(GOOD_PICTURE, "rb") as image_file: #open a real image in binary mode
        response = client.post("/prediction", data ={"file": (image_file, "sign.jpeg")}) #upload
    assert response.status_code == 200 #Page loaded OK
    assert ERROR_TEXT not in response.data #no error message there
    assert number_shown(response.data) #A digit appears


#Flask Routes: edge cases
def test_get_not_allowed_on_prediction(client):
    """Edge case: /prediction only accepts post"""
    assert client.get("/prediction").status_code == 405 # 405 = method not allowed

def test_unknown_page_returns_404(client):
    """Edge case: a page that does not exist returns 404"""
    assert client.get("/no-such-page").status_code == 404 # 404 = not found

def test_text_file_shows_error(client):
    """Edge case: a text file(not an image) will show an error message"""
    with open ("invalid_file.txt", "rb") as text_file: #the "not an image" file in the repo
        response = client.post(
            "/prediction", data={"file": (text_file, "invalid_file.txt")}
        )  # upload the text file
    assert ERROR_TEXT in response.data #error message shown
    assert not number_shown(response.data) #no fake predictions

def test_missing_file_shows_error(client):
    """Edge case: a request with no file shows the error message. """
    response = client.post("/prediction", data={}) #submit the form with nothing attached
    assert response.status_code == 200 #app does not crash
    assert ERROR_TEXT in response.data #it shows the error instead


#Image preprocessing and prediction
def test_predict_return_digit():
    """prediction on a real image gives a digit from 0 to 9"""
    digit = predict_result(preprocess_img(GOOD_PICTURE)) #run the full model on a real image
    assert 0 <= digit <= 9 #answer must be a valid single digit

def test_preprocess_resizes_anyshape():
    """any shape of image still becomes the format the model expects (1, 224, 224, 3)"""
    result = preprocess_img(make_image(size=(30, 90))) #weird shape
    assert result.shape == (1, 224, 224, 3) #1 image, 224x224, 3 colour channels

def test_preprocess_scales_pixels_to_0_1():
    """pixel values are scaled into teh range 0 - 1"""
    result = preprocess_img(make_image()) #regular colour image
    assert result.min() >= 0.0 #nothing under 0
    assert result.max() <= 1.0 #nothing above 1

def test_preprocess_grayscale_image():
    """a grayscale image (1 colour channel) will still work"""
    assert preprocess_img(make_image("L")).shape == (1, 224, 224, 3) # "L" = grayscale

def test_preprocess_transparent_image():
    """an image with transparency (4 channels, RGBA) will still work"""
    assert preprocess_img(make_image("RGBA")).shape == (1, 224, 224, 3) #RGBA is transparent
