import urllib

thisurl = "http://www-rohan.sdsu.edu/~gawron/index.html"

handle = urllib.urlopen(thisurl)

html_gunk = handle.read()

print html_gunk[:150]