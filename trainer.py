from detector import Detector

dt = Detector()
dt.train()

dt.save(dt.rf)
