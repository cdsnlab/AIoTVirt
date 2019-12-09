import matplotlib.pylab as plt

class drawgraph (object):
    def draw(self):
        self.id = 0

    def singlegraphfromdict(self, xydict,dirname, xlabel, ylabel, title):
        print ("drawing graph: ", title)
        plt.clf()
        plt.title(title)

        iterat = list(xydict.keys())
        val = list(xydict.values())
        print (iterat, val)
        plt.ylim(0,100)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)


        plt.plot(iterat,val)
        plt.grid(b=True, which='major', color='#999999', linestyle='-')

        plt.savefig(dirname+'/'+'single'+title+'.png')
        plt.show()

    def singlegraphefromcsv(self):
        print ('opening files to generate graph')