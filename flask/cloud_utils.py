import os

# To authenticate set the environment variable.
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "private/credentials.json"

def mnist_prediction(service, request):
    name = 'projects/{}/models/{}'.format("my-site-212022", "mnist_model")
    version = None
    if version is not None:
        name += '/versions/{}'.format(version)
    response = service.projects().predict(
        name=name,
        body=request
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])
    return response