from typing import TypeVar, Generic, Generator, Union, Tuple, Callable, List, Optional, Any, ForwardRef, ClassVar, TypeVarTuple, Unpack, overload

Tag = TypeVar("Tag")

TupleTypes = TypeVarTuple("TupleTypes")

BatchedTuples = List[Tuple[Unpack[TupleTypes]]]

FactsWithTag = Generator[Tuple[Tag, Tuple[*TupleTypes]], None, None]

FactsWithoutTag = Generator[Tuple[*TupleTypes], None, None]

# NOTE: This is a hack, the type would not check
Facts = FactsWithTag | FactsWithoutTag

BatchedFactsWithTag = List[List[Tuple[Tag, Tuple[*TupleTypes]]]]

BatchedFactsWithoutTag = List[List[Tuple[*TupleTypes]]]

# NOTE: This is a hack, the type would not check
BatchedFacts = BatchedFactsWithTag | BatchedFactsWithoutTag
