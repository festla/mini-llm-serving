from services.mini_serving.memory.continuous_allocator import (
    ContinuousAllocator
)



allocator = ContinuousAllocator(
    total_tokens=32
)


print("Initial")
allocator.show()



print("\nAllocate A(8)")
allocator.allocate(
    "A",
    8
)

allocator.show()



print("\nAllocate B(8)")
allocator.allocate(
    "B",
    8
)

allocator.show()



print("\nAllocate C(8)")
allocator.allocate(
    "C",
    8
)

allocator.show()



print("\nFree B")

allocator.free("B")

allocator.show()



print(
    "\nFree tokens:",
    allocator.free_tokens()
)


print(
    "Largest free block:",
    allocator.largest_free_block()
)


print(
    "Fragmentation ratio:",
    f"{allocator.fragmentation_ratio()*100:.2f}%"
)


print(
    "\nTry allocate D(10)"
)

result = allocator.allocate(
    "D",
    10
)


print(
    "Success:",
    result
)


allocator.show()
