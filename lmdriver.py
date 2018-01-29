import sys
import lmcorpus
import biaschecker_api
import lmlearning
import hashlib
import base64
import time
import os

biaschecker_url = os.getenv("BIAS_SERVER_URL", "http://localhost:3000")
biaschecker_app_id = os.getenv("BC_APP_ID","ebf263f37b8640f480dfc27ef324ffa0")
biaschecker_app_secret = os.getenv("BC_APP_SECRET","662e85a69b854f66b6245f6d9338331c")
UPDATE_QUEUE = True

def extract_keywords_from_next_item(queue_tag, update_queue):
	bc = biaschecker_api.BiasCheckerApi(biaschecker_url, biaschecker_app_id, biaschecker_app_secret)
	result = bc.get_next_article_in_queue(queue_tag)
	if(result == None):
		return None
	(articleId,data) = result
	if(data != None):
		hasher = hashlib.sha1(articleId.encode("utf-8"))
		fileId = base64.urlsafe_b64encode(hasher.digest()).decode("utf-8")
		lmcorpus.addToCorpus("articles", data, fileId)
		lml = lmlearning.LaughingMonkeyLearner()
		keywords = lml.getArticleSubjects(fileId)
		latestDate = lml.getArticleTimestamp(fileId)
		bc.replace_keywords(articleId, keywords)
	if(update_queue):
		bc.mark_visited(articleId, queue_tag)
	return (articleId, fileId, keywords)

def loop(update_queue):
	while(True):
		try:
			result = extract_keywords_from_next_item("keyword", update_queue)
			if(result != (None)):
				print(result) #keep track of processing for logs
			else:
				print("Nothing to process.  Sleeping for 1 second.")
		except Exception as e:
			print(e)
			pass
		time.sleep(1)

print(sys.version)
if(len(sys.argv) == 0):
	print("Driver was called with no arguments.")
else:
	print("Driver called with args.")
if __name__ == "__main__":
	loop(UPDATE_QUEUE)