
# import numpy as np
# from collections import deque

# def remove_duplicates_preserve_order(lst):
#     seen = set()
#     result = []
#     for item in lst:
#         if item not in seen:
#             seen.add(item)
#             result.append(item)
#     return result

# # class StimulateEnv(object):
    
# #     def __init__(self, user_id, users_dict, old_watched, newest_watched_video, state_size):
        
# #         self.users_dict = users_dict
# #         self.state_size = state_size 
# #         self.user_id = user_id

# #         # self.user_id, self.items, self.done = self.reset()
# #         self.users_history_lens = round(len(self.users_dict[self.user_id]) * 0.6)
# #         self.newest_watched_video = newest_watched_video
        
# #         # self.user_items = {data[0]:data[1] for data in self.users_dict[self.user_id]} #{'video_id': 'rated'}
# #         # self.items = [data[0] for data in self.users_dict[self.user_id][-self.state_size:]]
# #         # self.done = False 
# #         # old_watched = remove_duplicates_preserve_order(self.items) 
# #         old_watched = old_watched
# #         self.done_count = 3000


# class StimulateEnv(object):
    
#     """ 
#     this is for streaming phase

#     """
#     def __init__(self, user_id, newest_watched_video, users_dict, users_history_lens, state_size):
        
#         self.user_id = user_id
#         self.state_size = state_size 
#         self.users_dict = users_dict
#         self.users_history_lens = users_history_lens
        
#         # self.user_items = {data[0]:data[1] for data in self.users_dict[self.user_id]} #{'video_id': 'rated'}
#         self.items = [data[0] for data in self.users_dict[self.user_id][-self.state_size:]]
#         self.done = False 
#         self.old_watched = remove_duplicates_preserve_order(self.items) 
#         self.done_count = 3000
        
#     def step(self, recommend_item, newest_watched_video, old_watched):
        
#         reward = -0.5 
#         correctly_recommended = [] 
#         rewards = [] 
        
#         check_recomm_old = np.union1d(np.array(recommend_item), np.array(old_watched))
#         if newest_watched_video in recommend_item and len(check_recomm_old) == 0:
#             correctly_recommended.append(recommend_item)
#             rewards.append(1)
#         else:
#             rewards.append(-0.5)
        
#         deque_old_watched = deque(old_watched) 
#         deque_old_watched.append(newest_watched_video)
#         deque_old_watched.popleft()

#         if max(rewards) > 0: 
#             self.items = self.items[len(correctly_recommended):] + correctly_recommended 
        
#         reward = rewards
        
#         if len(old_watched) > self.done_count or len(old_watched) >= self.users_history_lens:
#             self.done = True
        
#         return self.items, reward, self.done, old_watched
    
#     def reset(self):
#         list_available_user = [key for key, values in self.users_dict.items()]
#         user = np.random.choice(list_available_user)
#         users_history_lens = round(len(self.users_dict[user]) * 0.6)
#         watched_videos = [video[0] for video in self.users_dict[user]][:users_history_lens]
#         done = False 
#         old_watched = remove_duplicates_preserve_order(watched_videos)
#         return user, old_watched[-self.state_size:], done
    
#     """ 
#     def reset(self):
#         self.user_id = np.random.choice(self.)
        
#     Too lazy, this part will serves only stimulation stage when RESET auto generate random user
#     and other information to start the algo
    



import numpy as np
from collections import deque

def remove_duplicates_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# class StimulateEnv(object):
    
#     def __init__(self, user_id, users_dict, old_watched, newest_watched_video, state_size):
        
#         self.users_dict = users_dict
#         self.state_size = state_size 
#         self.user_id = user_id

#         # self.user_id, self.items, self.done = self.reset()
#         self.users_history_lens = round(len(self.users_dict[self.user_id]) * 0.6)
#         self.newest_watched_video = newest_watched_video
        
#         # self.user_items = {data[0]:data[1] for data in self.users_dict[self.user_id]} #{'video_id': 'rated'}
#         # self.items = [data[0] for data in self.users_dict[self.user_id][-self.state_size:]]
#         # self.done = False 
#         # self.old_watched = remove_duplicates_preserve_order(self.items) 
#         self.old_watched = old_watched
#         self.done_count = 3000


class StimulateEnv(object):
    
    """ 
    this is for streaming phase

    """
    def __init__(self, users_dict, state_size, fix_user_id=None):

        self.users_dict = users_dict        
        self.state_size = state_size 
        self.num_history = 20
        
        self.users_history_lens = self.state_size 
        self.available_users = [i for i in self.users_dict.keys()] # self._generate_available_users()
        self.fix_user_id = fix_user_id
        self.user_id = fix_user_id if fix_user_id else np.random.choice(self.available_users)
        self.newest_watched_video = [video[0] for video in self.users_dict[self.user_id]][self.users_history_lens:][0]
        
        # self.user_items = {data[0]:data[1] for data in self.users_dict[self.user_id]} #{'video_id': 'rated'}
        self.items = [data[0] for data in self.users_dict[self.user_id][:self.state_size]][-self.num_history:]
        self.done = False 
        # self.old_watched = remove_duplicates_preserve_order(self.items) 
        self.old_watched = self.items
        self.done_count = 3000

    # def _generate_available_users(self):
    #     available_users = []
    #     for i, length in zip(self.users_dict.keys(), self.users_history_lens):
    #         # if length > self.state_size:
    #             available_users.append(i)
    #     return available_users
    
    def step(self, recommend_item):
        
        reward = -0.5 
        correctly_recommended = [] 
        rewards = [] 
        
        # print(self.newest_watched_video, recommend_item)
        if self.newest_watched_video in recommend_item: # and np.intersect1d(recommend_item, self.old_watched) == 0:
            corrected_item = np.intersect1d(self.newest_watched_video, recommend_item)
            correctly_recommended.append(corrected_item[0])
            rewards.append(1)
        else:
            rewards.append(-0.5)
        
        deque_old_watched = deque(self.old_watched) 
        deque_old_watched.append(self.newest_watched_video)
        deque_old_watched.popleft()

        # if max(rewards) > 0: 
        #     self.items = self.items[len(correctly_recommended):] + list(correctly_recommended)  
        # else: 
        self.items = self.items[1:] + [self.newest_watched_video]
        reward = rewards
        
        if len(self.old_watched) > self.done_count: #or len(self.old_watched) >= self.users_history_lens:
            self.done = True
        
        return self.items, reward, self.done, self.old_watched
    
    def reset(self):
        list_available_user = [key for key, values in self.users_dict.items()]
        user = np.random.choice(list_available_user)
        items = [video[0] for video in self.users_dict[user]][:self.num_history]
        done = False 
        old_watched = remove_duplicates_preserve_order(items)
        return user, old_watched, done
    
    """ 
    def reset(self):
        self.user_id = np.random.choice(self.)
        
    Too lazy, this part will serves only stimulation stage when RESET auto generate random user
    and other information to start the algo
    
    """
