"""Indexed binary min-heap with decrease-key.

Keys live in an external list (`keys[v]`, e.g. Dijkstra's dist array) so no key
duplication. `pos[v]` gives heap index, enabling O(log V) decrease-key.
Heap size is bounded by V (each vertex at most once), unlike a lazy-deletion heap.
"""


class IndexedMinHeap:
    def __init__(self, keys):
        self._keys = keys
        self._heap = []
        self._pos = [-1] * len(keys)

    def __len__(self):
        return len(self._heap)

    def __contains__(self, v):
        return self._pos[v] != -1

    def push_or_update(self, v):
        """Insert v, or restore order after keys[v] was DECREASED. O(log n)."""
        pos = self._pos
        if pos[v] == -1:
            self._heap.append(v)
            pos[v] = len(self._heap) - 1
        self._sift_up(pos[v])

    def pop_min(self):
        """Remove and return vertex with the smallest key. O(log n)."""
        heap, pos = self._heap, self._pos
        top = heap[0]
        last = heap.pop()
        pos[top] = -1
        if heap:
            heap[0] = last
            pos[last] = 0
            self._sift_down(0)
        return top

    def clear(self):
        """Empty the heap in O(current size) so the structure can be reused."""
        pos = self._pos
        for v in self._heap:
            pos[v] = -1
        self._heap.clear()

    def _sift_up(self, i):
        heap, pos, keys = self._heap, self._pos, self._keys
        v = heap[i]
        k = keys[v]
        while i > 0:
            p = (i - 1) >> 1
            pv = heap[p]
            if keys[pv] <= k:
                break
            heap[i] = pv
            pos[pv] = i
            i = p
        heap[i] = v
        pos[v] = i

    def _sift_down(self, i):
        heap, pos, keys = self._heap, self._pos, self._keys
        n = len(heap)
        v = heap[i]
        k = keys[v]
        while True:
            c = 2 * i + 1
            if c >= n:
                break
            r = c + 1
            if r < n and keys[heap[r]] < keys[heap[c]]:
                c = r
            cv = heap[c]
            if keys[cv] >= k:
                break
            heap[i] = cv
            pos[cv] = i
            i = c
        heap[i] = v
        pos[v] = i
