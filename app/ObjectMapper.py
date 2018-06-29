from app.Model import HistoryVO, PatientVO


def get_histories(histories):
    history_list = []
    for history in histories:  # a에서 안쪽 리스트를 꺼냄
        history = HistoryVO.HistoryVO(history[0], history[1], history[2], history[3], history[4], history[5], history[6])
        history_list.append(history.serialize())
    return history_list


def get_patient(patient_info):
    return PatientVO.PatientVO(patient_info)