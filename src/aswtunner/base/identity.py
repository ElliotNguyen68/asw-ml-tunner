from abc import ABC

class Identity(ABC):
    def __init__(self) -> None:
        super().__init__()

class TargetIdentity(Identity):
    def __init__(self,target:str) -> None:
        super().__init__()
        self.target = target

class RecommendIdentity(Identity):
    def __init__(self, user_identity:str, target_identity: str) -> None:
        super().__init__()
        self.user_identity=user_identity
        self.target_identity = target_identity