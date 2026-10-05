TOTAL_BLOCKS = 20

# =====================
# Continuous Allocation
# =====================

print("===== Continuous Memory =====")

memory = [None] * TOTAL_BLOCKS

def allocate_contiguous(name, size):

    for i in range(TOTAL_BLOCKS-size+1):

        if all(x is None for x in memory[i:i+size]):
            for j in range(i,i+size):
                memory[j]=name
            return i

    return -1


def free(name):

    for i in range(TOTAL_BLOCKS):
        if memory[i]==name:
            memory[i]=None



allocate_contiguous("A",5)
allocate_contiguous("B",6)
allocate_contiguous("C",4)

print(memory)

free("B")

print("After free B:")
print(memory)


print(
    "Allocate D(sieze=8):",
    allocate_contiguous("D",8)
)

print(memory)


# ================
# Paged Allocation
# ================

print("\n===== Paged Memory =====")

blocks=[None]*TOTAL_BLOCKS

def allocate_pages(name, num):

    ids=[]
    for i in range(TOTAL_BLOCKS):
        if blocks[i] is None:
            blocks[i] = name
            ids.append(i)

            if len(ids) == num:
                return ids

    return None


allocate_pages("A",5)
allocate_pages("B",6)
allocate_pages("C",4)

print(blocks)

# free B

for i,x in enumerate(blocks):
    if x=="B":
        blocks[i]=None

print("After free B:")
print(blocks)

print(
    "Allocate D:",
    allocate_pages("D",8)
)

print(blocks)










