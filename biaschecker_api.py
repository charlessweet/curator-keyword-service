import json
import dateutil.parser
import time
from urllib.request import Request, urlopen

class BiasCheckerApi(object):
	__article_list = None
	__biaschecker_server_url = None
	__biaschecker_app_id = None
	__biaschecker_app_secret = None
	
	def __init__(self, server, biaschecker_app_id, biaschecker_app_secret):
		self.__biaschecker_server_url = server
		self.__biaschecker_app_id = biaschecker_app_id
		self.__biaschecker_app_secret = biaschecker_app_secret
		
	def call_biaschecker(self, relative_url, method, data = None):
		url = self.__biaschecker_server_url + relative_url;
		req = Request(url)
		req.add_header("content-type", "application/json")
		req.add_header("X-BIASCHECKER-API-KEY", self.__biaschecker_app_secret)
		req.add_header("X-BIASCHECKER-APP-ID", self.__biaschecker_app_id)
		req.get_method = lambda:method
		if(data != None):
			req.data = json.dumps(data).encode('ascii');
		json_data = urlopen(req).read()
		json_obj = json.loads(json_data)
		return json_obj

	def mark_visited(self, article_id, queue_tag):
		relative_url = "/articles/" + article_id + "/tags/" + queue_tag;
		r = self.call_biaschecker(relative_url, "PUT");
		return r

	def replace_keywords(self, article_id, keywords):
		keyword_url = "/articles/" + article_id + "/keywords"
		json_obj = self.call_biaschecker(keyword_url, "PUT", keywords)
		return json_obj
	
	def get_next_article_in_queue(self, queue_tag):
		article_id_url = "/articles?limit=1&missing_tag=" + queue_tag
		json_obj = self.call_biaschecker(article_id_url, "GET")
		if(len(json_obj["rows"]) == 0):
			return None
		row = json_obj["rows"][0];
		article_id = row["id"]

		#get article text
		article_body_url = "/articles/" + article_id
		json_obj = self.call_biaschecker(article_body_url,"GET")
		if(len(json_obj["rows"]) == 0):
			return None
		row = json_obj["rows"][0]
		text = [row["value"] for row in json_obj["rows"]]
		return (article_id,text[0])
	
	def get_article_text(self, article_id, created):
		article_body_url = "/articles/" + article_id
		json_obj = self.call_biaschecker(article_body_url,"GET")
		if(len(json_obj["rows"]) == 0):
			return None
		text = [row["value"] for row in json_obj["rows"]]
		dt = dateutil.parser.parse(created)
		timestamp = str(time.mktime(dt.timetuple()))
		return (timestamp, text[0])