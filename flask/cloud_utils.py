import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "private/credentials.json"

def mnist_prediction(service, request):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
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