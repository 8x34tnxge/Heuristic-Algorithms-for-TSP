from typing import Any, Dict, Iterator, List, Union


class DataLoader(list):
    def __init__(self, data: Union[List[Any], Dict[Any, Any]]) -> None:
        self.data: Union[List[Any], Dict[Any, Any]] = data
        if isinstance(self.data, dict):
            self.keys = list(data.keys())
        else:
            self.keys = [x for x in range(len(data))]
        self.iterator = DataIterator(self.data, self.keys)

    def __getitem__(self, pos: int) -> Any:
        try:
            value = self.data[pos]
        except:
            raise KeyError(f"The key {pos} is not included in data")
        return value

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Any]:
        return self.iterator


class DataIterator(Iterator):
    def __init__(self, data: Union[List[Any], Dict[Any, Any]], keys: List[Any]) -> None:
        self.current = 0
        self.data = data
        self.keys = keys

    def __iter__(self) -> Iterator:
        return super().__iter__()

    def __next__(self) -> Any:
        if self.current < len(self.keys):
            key = self.keys[self.current]
            self.current += 1
            return key, self.data[key]
        self.current = 0
        raise StopIteration
