class PatientVO:
    def __init__(self, user_info):
        self.patient_id = user_info[0]
        self.patient_name = user_info[1]
        self.patient_info = user_info[2]
