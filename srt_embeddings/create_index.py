import os
import lancedb


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    db_path = os.path.join(current_dir, "transcripts_lancedb")
    db = lancedb.connect(db_path)

    table_name = "transcripts"
    transcripts_table = db.open_table(table_name)
    transcripts_table.create_scalar_index("id", replace=True)
    transcripts_table.create_fts_index("text", replace=True, use_tantivy=False)
    transcripts_table.create_index(metric='cosine', vector_column_name='embedding')
    # transcripts_table.create_index(metric='cosine', vector_column_name='multivector_embedding')
    






if __name__ == '__main__':
    main()