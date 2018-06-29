class HistoryVO:
    def __init__(self, patient_id, start_time, end_time, handover, longitude, latitude):
        try:
            self.patient_id = patient_id
            self.start_time = start_time
            self.end_time = end_time
            self.handover = handover
            self.longitude = longitude
            self.latitude = latitude
        except Exception as e:
            self.patient_id = None
            self.start_time = None
            self.end_time = None
            self.handover = None
            self.longitude = None
            self.latitude = None

    def serialize(self):
        return{
            'patientId': self.patient_id,
            'startTime': self.start_time,
            'endTime': self.end_time,
            'handover': self.handover,
            'longitude': self.longitude,
            'latitude':self.latitude
        }

