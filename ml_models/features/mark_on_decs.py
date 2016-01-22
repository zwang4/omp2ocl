import os, sys

apps=["bt", "ep", "is", "mg", "lu", "sp", "ft", "cg"]
dataset=["S", "W", "A", "B", "C"]

class radeon_tree:
	def __init__(self, features):
		self.f1 = float(features[0])
		self.f2 = float(features[1])
		self.f3 = float(features[2])
		self.f4 = float(features[3])

	def doIt(self):
		if (self.f1 < 0.03) == False:
			if (self.f3 < 3300) == False:
				return "A2"
			else:
				return "A1"
		else:
			if (self.f4 < 7.65):
				if (self.f3 < 21):
					if (self.f2 < 0.99):
						return "A5"
					else:
						return "A6"
				else:
					return "A3"
			else:
				if (self.f3 < 0.02):
					if (self.f4 < 134):
						if (self.f4 < 30):
							return "A8"
						else:
							return "A9"
					else:
						return "A7"

				else:
					return "A4"

class nvidia_tree:
	def __init__(self, features):
		self.f1 = float(features[0])
		self.f2 = float(features[1])
		self.f3 = float(features[2])
		self.f4 = float(features[3])

	def doIt(self):
		if (self.f1 < 0.69) == False:
			return "A1"
		if (self.f4 < 12) == False:
			return "A2"
		if (self.f3 < 29) == False:
			return "A3"
		if (self.f2 < 0.70) == False:
			if (self.f3 < 0.05):
				if (self.f4 < 1.40):
					if (self.f2 < 0.9):
						return "A10"
					else:
						if (self.f1 < 1.4):
							return "A12"
						else:
							return "A11"
				else:
					return "A7"
			else:
				if (self.f4 < 1.40):
					return "A9"
				else:
					return "A8"
		else:
			if (self.f4 < 6.50):
				if (self.f3 < 9):
					return "A5"
				else:
					return "A6"
			else:
				return "A4"


class mark_desc_tree:
	def __init__(self, platform, dct):
		self.platform = platform
		self.d = dct

	def process_app(self, app):
		fn = dct + "/" + app + "_test.csv"
		f = open(fn, "r")
		fv=[]
		for line in f:
			line=line.replace('\n', '')
			features=line.split(',')
			fv.append(features)
		f.close()
		res=[]
		for i in range(0, len(fv)):
			ap_name = app + "." + dataset[i]
			feature=fv[i]
			if (self.platform == "NVIDIA"):
				m = nvidia_tree(feature)
			if (self.platform == "RADEON"):
				m = radeon_tree(feature)
			c = m.doIt()
			res.append([ap_name, c])
		return res
		
	def doIt(self):
		res=[]
		cls=dict()
		for a in apps:
			r=self.process_app(a)
			for app, rc in r:
				res.append([app, rc])

		print res
		for app, rc in res:
			if rc in cls.keys():
				d=cls[rc]
			else:
				d=[]
			d.append(app)
			cls[rc]=d

		f=open(self.platform + ".map", "w")
		for rc in cls.keys():
			f.write(rc + ":")
			d=cls[rc]
			for di in d:
				f.write(di + ",")
			f.write("\n")
		f.close()

dct=sys.argv[1]
platform=sys.argv[2]
m = mark_desc_tree(platform, dct)
m.doIt()
