import tweepy
#import simplejson as json
from tweepy.models import Status
import time

tweetcount = 0
filename = ''

def gen_file_name():
    timestamp = int(time.time() * 1000)
    #filename = '/data/shared/twitter/twitter_stream_data2' + str(timestamp) + '.lines'
    filename = str(timestamp) + '.lines'
    return filename

def my_on_data(data):
    global filename, tweetcount
    if data.startswith('{') and 'status_id' in data:
        try:
            timestamp = int(time.time() * 1000)
            f = open(filename, 'a')
            f.write(str(timestamp) + '\t' + str(data) + '\n')
            f.close()
            tweetcount += 1
            if tweetcount % 100 == 0:
                print tweetcount
            if tweetcount % 5000 == 0:
                # print tweetcount
                #f = open(filename + '.unlock', 'w')
                #f.close()
                filename = gen_file_name()
        except:
            raise
            # pass
            
def my_on_error(status_code):
    print 'Error: ', str(status_code)


def main():
    global tweetcount, filename
    auth = tweepy.OAuthHandler('hTo3lEHIVIP23wr4FfkRiCnEk','g1N56BCPf0OoMARJpoSB2pY0skNLSd6nbD8Z83inI9cPGI2EwJ') # consumerkey, consumersecret
    auth.set_access_token('1429600495-K5FmRweMpahWufn6LD7KsZ9kluobcUmxZbFip8V','BJcmcJMvoqs1DPzfl0i2JSp7j19kCfakdaVVgBGUhKZz5') # accesstoken, accesstokensecret
    api = tweepy.API(auth)
    
    tweetcount = 0
    filename = gen_file_name() 

    streamlistener = tweepy.StreamListener(api)
    streamlistener.on_data = my_on_data   #callback function
    streamlistener.on_error = my_on_error
    stream = tweepy.Stream(auth, listener=streamlistener, secure=True)
    

    # start the stream filtering (an infinite loop)
#    stream.filter(track=['iphone 5'])

    try:
	#print 'sampling'
	stream.sample()
    except:
	#f = open(filename + '.unlock', 'w')
	#f.close()
	#filename = gen_file_name()
	raise

if __name__ == '__main__':
    main()
