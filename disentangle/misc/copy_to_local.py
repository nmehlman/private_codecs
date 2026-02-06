import os
import subprocess


def copy_files_from_slurm(remote_user, remote_host, log_dir, local_dir, N):
    """
    Copies the first N .pkl files from the SLURM cluster log directory to the local directory.
    
    Parameters:
    remote_user (str): The username for the remote SLURM cluster.
    remote_host (str): The hostname or IP address of the remote SLURM cluster.
    log_dir (str): The directory on the SLURM cluster where the .pkl files are located.
    local_dir (str): The local directory to copy the files to.
    N (int): The number of files to copy.
    """
    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)

    # Command to list .pkl files on the remote server
    command = f'ssh {remote_user}@{remote_host} "ls {log_dir}/eval/*.pkl"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Split the output into a list of files
    files = result.stdout.splitlines()
    files.sort(key=lambda x: int(x.split('/')[-1].split('_')[0]))  # Sort files to ensure consistent ordering
    print(f'Found {len(files)} .pkl files in {log_dir}.')
    # Copy the first N files using scp
    for file in files[:N]:
        file_name = os.path.basename(file)
        src = f'{remote_user}@{remote_host}:{os.path.join(log_dir,"eval", file_name)}'
        dst = os.path.join(local_dir, file_name)
        subprocess.run(['scp', src, dst])
        print(f'Copied {file_name} to {local_dir}')


if __name__ == "__main__":

    # Example usage
    remote_user = 'nmehlman'
    remote_host = 'discovery.usc.edu'
    log_dir = '/project2/shrikann_35/nmehlman/logs/private_codecs/tensorboard/longer_training/version_0/'
    local_dir = '/Users/nick/Desktop/Private Codecs/tests'
    N = 3  # Number of files to copy

    copy_files_from_slurm(remote_user, remote_host, log_dir, local_dir, N)