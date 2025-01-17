import requests

def post_text_message_to_slack(text, slack_channel = "slack_dummy_channel"):
  
        slack_token = 'abc'
        credentials_dictionary = {'token': slack_token,
                                  'channel': slack_channel,
                                  'text': text,
                                  'blocks': None}

        return requests.post('https://slack.com/api/chat.postMessage', credentials_dictionary).json()