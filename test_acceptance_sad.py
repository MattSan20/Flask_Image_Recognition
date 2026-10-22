# test_acceptance_sad.py


def test_acceptance_missing_file(client):
    """Test the prediction route with a missing file"""
    # Simulate a POST request with no file data
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    # 1. Ensure the response status code is 200, indicating the request was processed.
    assert response.status_code == 200
    #    Modify the message check if your application uses a different error response text.
    assert b"File cannot be processed" in response.data  # Expected error message
