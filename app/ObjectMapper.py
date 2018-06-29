from app.Model import HistoryVO

def get_histories(histories):
    history_list = []
    for i in histories:  # a에서 안쪽 리스트를 꺼냄
        history = HistoryVO.HistoryVO(histories[0], histories[1], histories[2], histories[3], histories[4], histories[5])
        history_list.append(history.serialize())
    return history_list