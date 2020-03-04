import os
import pandas as pd
from getpass import getuser
import matplotlib.pyplot as plt

FOLDER = 'mountaincar'



 # Folder name for good ol' Windows
dirname = os.path.dirname(__file__)
out_folder = os.path.join(dirname, '../models/'+FOLDER+'/')
username = getuser()
filename = out_folder + username + '-' + 'logs.csv'
plot_folder=os.path.join(dirname,'../plots')

df = pd.read_csv(filename, header=0, names=['Date', 'Generation', 'Mean', 'Min', 'Max'])

ax = plt.gca()
df.plot(kind='line', x='Generation', y='Mean', ax=ax)
df.plot(kind='line', x='Generation', y='Min', ax=ax)
df.plot(kind='line', x='Generation', y='Max', ax=ax)
plt.ylabel('Fitness')
plt.title(FOLDER.capitalize()+' environment')

if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)

plt.savefig(plot_folder +'/'+FOLDER+'.png')


