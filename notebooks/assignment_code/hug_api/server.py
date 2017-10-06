import os
import json
import hug
import pandas as pd
from sklearn.externals import joblib

#app = hug.API(__name__)

@hug.post('/predict')
def apicall(body):

	"""Converting json to pandasdataframe
	"""
	try:
		otherdf = pd.read_json(body, orient='split')

		print("The shape of the received file is {}".format(otherdf.shape))
	except Exception as e:
		raise e

	if otherdf.empty:
		return(bad_request())
	else:
		"""Loading the pickled model using joblib 
		"""
		clf = 'model2.pk'
		
		loaded_model = joblib.load(os.getcwd()+'/models/'+str(clf))
		predictions = loaded_model.predict(otherdf.as_matrix())

		"""An ndarray is not json serializable, so converting to pandas.Series
		"""
		predictions_series = pd.Series(predictions)
		response = json.dumps({'predictions':predictions_series.to_json()})

		return(response)


@hug.exception(Exception)
def bad_request(exception):
	message = {
			'status': 400,
			'message': 'Bad Request: Please check your data payload...',
	}
	resp = json.dumps(message)

	return resp