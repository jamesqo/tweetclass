

fin = open("full-corpus.csv", "r")
fout = open("output.csv", "w+")
lines = tuple(fin)

hash = {}
hash['"Sentiment"'] = 0
hash['"positive"'] = 0
hash['"irrelevant"'] = 0
hash['"negative"'] = 0
hash['"neutral"'] = 0

for line in lines:
    line = line.split(",")
    if len(line) >= 2 and line[1] in hash:
        hash[line[1]] += 1
    #hash[line[1]] += 1 if line[1] in hash else hash[line[1]] = 0
    if len(line) >= 4 and line[1] in hash and  hash[line[1]] <= 200: fout.write("|"+ line[1].replace('"','')+"|,|"+ line[4].replace('"','').replace('\n','') + "|\n")


fout.close()
