import scallopy
import os

# rel variable = {(1,), (2,), (3,)}
# rel time_stamp = {(1, 1584)}
# rel time_stamp_ct = {(1,)}
# rel positive_binary_atom = {(1581, "throwing", -1, -2, 0), (1582, "throwing", -3, -2, 1581), (1583, "toward", -2, -1, 1582), (1584, "toward", -2, -3, 1583)}

if __name__ == "__main__":
    
    provenance = "topkproofs"
    k = 5
    common_scl_path = os.path.abspath(os.path.join(os.path.abspath(__file__), '../P14_06.scl'))
    scallop_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    scallop_ctx.import_file(common_scl_path)
        
    program = "Finally(Logic(Binary(\"picking\",Const(-2),Const(-1))))"
    # scl_dict = {'spec': program}
    scallop_ctx.add_facts("spec", [(1.0, (program,))])
    scallop_ctx.run()
    
    print('here')
    
