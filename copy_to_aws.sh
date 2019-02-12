scp -v -i "~/Documents/Do Not Touch/aws_key.pem" -r "/Users/sahni/Documents/Code/github_repos/CMUCourseWork/11785/HW1-Part2/Code/dataset.py" ubuntu@ec2-18-221-198-162.us-east-2.compute.amazonaws.com:~/sahni/HW1-Part2/Code/
scp -v -i "~/Documents/Do Not Touch/aws_key.pem" -r "/Users/sahni/Documents/Code/github_repos/CMUCourseWork/11785/HW1-Part2/Code/main.py" ubuntu@ec2-18-221-198-162.us-east-2.compute.amazonaws.com:~/sahni/HW1-Part2/Code/
scp -v -i "~/Documents/Do Not Touch/aws_key.pem" -r "/Users/sahni/Documents/Code/github_repos/CMUCourseWork/11785/HW1-Part2/Code/model.py" ubuntu@ec2-18-221-198-162.us-east-2.compute.amazonaws.com:~/sahni/HW1-Part2/Code/

# Copy Dataset Onetime Only.
#scp -v -i "~/Documents/Do Not Touch/aws_key.pem" -r "/Users/sahni/Documents/Code/github_repos/CMUCourseWork/11785/HW1-Part2/Data/dev.pt" ubuntu@ec2-3-17-172-51.us-east-2.compute.amazonaws.com:~/sahni/HW1-Part2/Data/
#scp -v -i "~/Documents/Do Not Touch/aws_key.pem" -r "/Users/sahni/Documents/Code/github_repos/CMUCourseWork/11785/HW1-Part2/Data/dev_labels.pt" ubuntu@ec2-3-17-172-51.us-east-2.compute.amazonaws.com:~/sahni/HW1-Part2/Data/
#scp -v -i "~/Documents/Do Not Touch/aws_key.pem" -r "/Users/sahni/Documents/Code/github_repos/CMUCourseWork/11785/HW1-Part2/Data/dev_utteranceIndices.pt" ubuntu@ec2-3-17-172-51.us-east-2.compute.amazonaws.com:~/sahni/HW1-Part2/Data/
#scp -v -i "~/Documents/Do Not Touch/aws_key.pem" -r "/Users/sahni/Documents/Code/github_repos/CMUCourseWork/11785/HW1-Part2/Data/test.pt" ubuntu@ec2-3-17-172-51.us-east-2.compute.amazonaws.com:~/sahni/HW1-Part2/Data/
#scp -v -i "~/Documents/Do Not Touch/aws_key.pem" -r "/Users/sahni/Documents/Code/github_repos/CMUCourseWork/11785/HW1-Part2/Data/test_utteranceIndices.pt" ubuntu@ec2-3-17-149-35.us-east-2.compute.amazonaws.com:~/sahni/HW1-Part2/Data/
#scp -v -i "~/Documents/Do Not Touch/aws_key.pem" -r "/Users/sahni/Documents/Code/github_repos/CMUCourseWork/11785/HW1-Part2/Data/train.pt" ubuntu@ec2-3-17-172-51.us-east-2.compute.amazonaws.com:~/sahni/HW1-Part2/Data/
#scp -v -i "~/Documents/Do Not Touch/aws_key.pem" -r "/Users/sahni/Documents/Code/github_repos/CMUCourseWork/11785/HW1-Part2/Data/train_labels.pt" ubuntu@ec2-3-17-172-51.us-east-2.compute.amazonaws.com:~/sahni/HW1-Part2/Data/
#scp -v -i "~/Documents/Do Not Touch/aws_key.pem" -r "/Users/sahni/Documents/Code/github_repos/CMUCourseWork/11785/HW1-Part2/Data/train_utteranceIndices.pt" ubuntu@ec2-3-17-172-51.us-east-2.compute.amazonaws.com:~/sahni/HW1-Part2/Data/
