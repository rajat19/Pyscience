import csv, random
raw, smackdown, nxt, legends = [], [], [], []
with open('datasets/wwe/draft.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	abv90, abv85, rest = 0, 0, 0
	total = []
	wc, sc, esc, ttc, top, sec, newlist = [], [], [], [], [], [], []
	for row in reader:
		total.append(row)
	# random.shuffle(total)
	# random.shuffle(total)
	for row in total:
		if len(row) == 3 and row[2] == 'wc':
			wc.append(row[:1])
		elif len(row) == 3 and row[2] == 'sc':
			sc.append(row[:1])
		elif len(row) == 3 and row[2] == 'esc':
			esc.append(row[:1])
		elif len(row) == 3 and row[2] == 'ttc':
			ttc.append(row[:1])
		elif (int)(row[1]) >= 90:
			top.append(row[:1])
		elif (int)(row[1]) >= 85 and len(top) < 40:
			top.append(row[:1])
		elif (int)(row[1]) >= 85 and len(sec) < 40:
			sec.append(row[:1])
		elif (int)(row[1]) >= 84 and len(sec) < 40:
			sec.append(row[:1])
		else: newlist.append(row[:1])
	for i in range(5):
		random.shuffle(wc)
		random.shuffle(sc)
		random.shuffle(ttc)
		random.shuffle(top)
		random.shuffle(sec)
		random.shuffle(newlist)
	raw = [wc[0], sc[0], ttc[0], ttc[1]] + top[:10] + sec[:10] + newlist[:11]
	smackdown = [wc[1], sc[1], ttc[2], ttc[3]] + top[10:20] + sec[10:20] + newlist[11:22]
	nxt = [wc[2], esc[0], esc[1], esc[2]] + top[20:30] + sec[20:30] + newlist[22:33]
	legends = [wc[3], sc[2], ttc[4], ttc[5]] + top[30:] + sec[30:] + newlist[33:]
	
with open('datasets/wwe/finaldraft.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	for i in range(34):
		writer.writerow(raw[i] + smackdown[i] + nxt[i] + legends[i])
	print('Done')