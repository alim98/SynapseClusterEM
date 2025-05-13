Initial Connectome Loading (One-Time, Partial):
The sample_connectome() function loads synapse positions and agglomeration IDs from a connectome HDF5 file
This happens only once using global variables positions and agglo_ids ( line 70)
Only the position data is loaded, not the actual volumetric data
This data is then filtered to include only positions within designated "plexiform bboxes" (a specific region of interest)

---
Batch-Based Sampling from Connectome:
After the initial load, the system samples a batch of positions randomly from the connectome dataset (specified by batch_size)
This sample is a small subset of the full dataset (5-10 samples at a time)

---
For each position in the batch, the system loads the actual volumetric data one at a time in a sequential loop:
     for position, src_agglo_id in wrapper(zip(positions, agglo_ids)):
         r, m = get_synapse_data(position, src_agglo_id)
         raw.append(r)
         mask.append(m)
This process is not parallel - each position is processed one after another

-----
WebKnossos Access.
For each position, the system Opens a connection to WebKnossos
Extracts 3D bounding box (80x80x80) around the synapse
Gets both raw EM data and agglomeration data
Closes the connection
This is done sequentially for each sample in the batch
------
