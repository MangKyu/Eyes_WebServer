class PatientVO:
    def __init__(self, user_info):
        try:
            self.patient_id = user_info[0]
            self.patient_name = user_info[1]
            self.patient_info = user_info[2]
            self.patient_image = user_info[3]
        except Exception as e:
            self.patient_id = None
            self.patient_name = None
            self.patient_info = None
            self.patient_image = None

    def serialize(self):
        return{
            'patientId': self.patient_id,
            'patientName': self.patient_name,
            'patientInfo': self.patient_info,
            'patientImage': self.patient_image
        }