def getStudentistAfterIgnoring(students, students_to_be_ignored):
    """
    @param students: A list of students that are to be considered. Has to be integer IDs
    @param students_to_be_ignored: List of students to be ignored.
    @return: Final processed list of students.
    """
    for e in students_to_be_ignored:
        if e in students:
            students.remove(e)

    return students


def getStudentsFromFolderNames(prefix, folder_names):
    """

    @param prefix: Prefix to be removed
    @param folder_names: Folder name of the students whos prefix are to be removed.
    @return: Student Names as int.
    """
    result = []
    for folder_name in folder_names:
        if folder_name.startswith(prefix):
            try:
                result.append(int(folder_name[len(prefix):]))
            except ValueError:
                print("Student ID couldn't be converted to Integer!")

    return result
