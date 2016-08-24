fin = open("full-corpus.csv", "r")
fout = open("output.csv", "w+")
lines = tuple(fin)

for line in lines:
  line = line.split(",")
  if len(line) >= 4: fout.write("|"+ line[1].replace('"','')+"|,|"+ line[4].replace('"','').replace('\n','') + "|" + " \n")

fout.close()
